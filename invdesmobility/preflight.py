from __future__ import annotations

import os

from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values
from pymongo import MongoClient
import requests

from .models import PreflightResult, RunSource


class PreflightError(RuntimeError):
    pass


_REQUIRED_ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "MOBILITY_DB_URI",
    "MONGO_URI",
)

_SUPPORTED_OPENAI_COMPATIBLE_HOSTS = {
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "dashscope-us.aliyuncs.com",
    "openrouter.ai",
}


def load_effective_env(project_root: str | os.PathLike[str]) -> dict[str, str]:
    root = Path(project_root)
    merged: dict[str, str] = {}
    for candidate in (root / ".env", root / ".env.local"):
        if not candidate.exists():
            continue
        values = dotenv_values(candidate)
        for key, value in values.items():
            if not key or value is None:
                continue
            merged[str(key)] = str(value)
    for key, value in os.environ.items():
        merged[str(key)] = str(value)
    return merged


def _is_placeholder(value: str | None) -> bool:
    text = str(value or "").strip()
    return (not text) or ("__FILL_ME__" in text)


def ensure_mongo_connectivity(mongo_uri: str) -> None:
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
    )
    try:
        client.admin.command("ping")
    except Exception as exc:
        raise PreflightError(f"MongoDB connectivity check failed for MONGO_URI={mongo_uri!r}: {exc}") from exc
    finally:
        try:
            client.close()
        except Exception:
            pass


def _bool_env(value: str | None, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _hostname_for_base_url(base_url: str | None) -> str:
    text = str(base_url or "").strip()
    if not text:
        return ""
    try:
        return str(urlparse(text).hostname or "").strip().lower()
    except Exception:
        return ""


def _is_dashscope_host(hostname: str) -> bool:
    return hostname in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}


def ensure_supported_llm_endpoints(effective_env: dict[str, str]) -> None:
    provider = str(effective_env.get("LLM_PROVIDER") or "").strip().lower()
    if provider not in {"openai", "openai_compatible"}:
        raise PreflightError(
            "Only OpenAI-compatible Qwen/DashScope and OpenRouter runtimes are enabled for loop launches; "
            f"got LLM_PROVIDER={provider!r}."
        )

    llm_base_url = str(effective_env.get("LLM_BASE_URL") or "").strip()
    embedding_base_url = str(effective_env.get("EMBEDDING_BASE_URL") or llm_base_url).strip()
    for label, base_url in (("LLM_BASE_URL", llm_base_url), ("EMBEDDING_BASE_URL", embedding_base_url)):
        hostname = _hostname_for_base_url(base_url)
        if hostname not in _SUPPORTED_OPENAI_COMPATIBLE_HOSTS:
            raise PreflightError(
                f"{label} must point to DashScope/Qwen or OpenRouter for loop launches, got {base_url!r}."
            )

    llm_hostname = _hostname_for_base_url(llm_base_url)
    llm_model = str(effective_env.get("LLM_MODEL") or "").strip().lower()
    if _is_dashscope_host(llm_hostname) and not llm_model.startswith("qwen"):
        raise PreflightError(
            f"DashScope loop launches require a Qwen chat model, got LLM_MODEL={effective_env.get('LLM_MODEL')!r}."
        )


def ensure_embedding_connectivity(effective_env: dict[str, str]) -> None:
    rag_required = _bool_env(effective_env.get("RAG_REQUIRED"), True)
    agentic_policy_enabled = _bool_env(effective_env.get("AGENTIC_POLICY_ENABLED"), True)
    if not rag_required and not agentic_policy_enabled:
        return

    embedding_model = str(effective_env.get("EMBEDDING_MODEL") or "").strip()
    embedding_base_url = str(effective_env.get("EMBEDDING_BASE_URL") or effective_env.get("LLM_BASE_URL") or "").strip()
    embedding_api_key = str(effective_env.get("EMBEDDING_API_KEY") or effective_env.get("LLM_API_KEY") or "").strip()
    if _is_placeholder(embedding_model) or _is_placeholder(embedding_base_url) or _is_placeholder(embedding_api_key):
        raise PreflightError("Embedding configuration is incomplete for full-autonomy RAG.")

    url = embedding_base_url.rstrip("/") + "/embeddings"
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {embedding_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": embedding_model,
                "input": "invdesmobility embedding probe",
            },
            timeout=30,
        )
    except Exception as exc:
        raise PreflightError(
            f"Embedding connectivity check failed for EMBEDDING_MODEL={embedding_model!r} via {embedding_base_url!r}: {exc}"
        ) from exc

    try:
        payload = dict(response.json() or {})
    except Exception as exc:
        raise PreflightError(
            f"Embedding connectivity check returned non-JSON response for EMBEDDING_MODEL={embedding_model!r}: "
            f"status={response.status_code}"
        ) from exc

    error = dict(payload.get("error") or {})
    if error:
        message = str(error.get("message") or payload)
        code = error.get("code")
        raise PreflightError(
            f"Embedding connectivity check failed for EMBEDDING_MODEL={embedding_model!r} via {embedding_base_url!r}: "
            f"{message} (code={code}, http_status={response.status_code})"
        )

    data = list(payload.get("data") or [])
    first = dict(data[0] or {}) if data else {}
    embedding = first.get("embedding")
    if not data or not isinstance(embedding, list) or not embedding:
        raise PreflightError(
            f"Embedding connectivity check returned no usable embedding data for EMBEDDING_MODEL={embedding_model!r} "
            f"via {embedding_base_url!r}."
        )


def run_preflight(
    *,
    source: RunSource,
    two_d_mobility_root: str | os.PathLike[str],
    expected_cif_count: int | None,
) -> PreflightResult:
    root = Path(two_d_mobility_root)
    env_local_path = root / ".env.local"
    batch_script = root / "run_mongo_batch.py"

    if not root.exists():
        raise PreflightError(f"2d-mobility repository root does not exist: {root}")
    if not env_local_path.exists():
        raise PreflightError(f"Missing 2d-mobility .env.local: {env_local_path}")
    if not batch_script.exists():
        raise PreflightError(f"Missing 2d-mobility batch entrypoint: {batch_script}")
    if not source.items:
        raise PreflightError(f"No ranked CIF files were resolved from source run: {source.run_link}")
    if expected_cif_count is not None and len(source.items) != expected_cif_count:
        raise PreflightError(
            f"Expected exactly {expected_cif_count} ranked CIF files under {source.top10_dir}, got {len(source.items)}"
        )
    if not Path(source.top10_csv_path).exists():
        raise PreflightError(f"Missing top10 CSV: {source.top10_csv_path}")
    if not Path(source.manifest_path).exists():
        raise PreflightError(f"Missing manifest JSON: {source.manifest_path}")

    effective_env = load_effective_env(root)
    missing = [key for key in _REQUIRED_ENV_KEYS if _is_placeholder(effective_env.get(key))]
    if missing:
        raise PreflightError(
            "2d-mobility effective environment is missing required settings: " + ", ".join(missing)
        )

    ensure_supported_llm_endpoints(effective_env)
    ensure_mongo_connectivity(effective_env["MONGO_URI"])
    ensure_embedding_connectivity(effective_env)
    return PreflightResult(
        two_d_mobility_root=str(root),
        effective_env=effective_env,
    )
