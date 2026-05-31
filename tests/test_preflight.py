from __future__ import annotations

import unittest

from unittest.mock import patch

from invdesmobility.preflight import PreflightError, ensure_embedding_connectivity, ensure_supported_llm_endpoints


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return dict(self._payload)


class PreflightTests(unittest.TestCase):
    def test_embedding_connectivity_accepts_valid_embedding_payload(self) -> None:
        env = {
            "LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "LLM_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-v4",
        }
        with patch("invdesmobility.preflight.requests.post", return_value=_FakeResponse({"data": [{"embedding": [0.1, 0.2]}]})):
            ensure_embedding_connectivity(env)

    def test_supported_llm_endpoints_accepts_dashscope_qwen(self) -> None:
        ensure_supported_llm_endpoints(
            {
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "LLM_MODEL": "qwen3.6-plus",
                "EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
        )

    def test_supported_llm_endpoints_accepts_openrouter_fallback(self) -> None:
        ensure_supported_llm_endpoints(
            {
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "https://openrouter.ai/api/v1",
                "LLM_MODEL": "minimax/minimax-m2.7",
                "EMBEDDING_BASE_URL": "https://openrouter.ai/api/v1",
            }
        )

    def test_supported_llm_endpoints_rejects_removed_provider_hosts(self) -> None:
        with self.assertRaisesRegex(PreflightError, "DashScope/Qwen or OpenRouter"):
            ensure_supported_llm_endpoints(
                {
                    "LLM_PROVIDER": "openai",
                    "LLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
                    "LLM_MODEL": "glm-4.5",
                    "EMBEDDING_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
                }
            )

    def test_embedding_connectivity_raises_on_router_error_payload(self) -> None:
        env = {
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "openai/text-embedding-3-large",
        }
        with patch(
            "invdesmobility.preflight.requests.post",
            return_value=_FakeResponse({"error": {"message": "No successful provider responses.", "code": 404}}),
        ):
            with self.assertRaisesRegex(PreflightError, "No successful provider responses"):
                ensure_embedding_connectivity(env)


if __name__ == "__main__":
    unittest.main()
