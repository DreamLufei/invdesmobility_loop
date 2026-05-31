#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SECRETS_ENV="${LOOP_LOCAL_SECRETS_ENV:-${INVDESMOBILITY_LOCAL_SECRETS_ENV:-${HOME}/.config/invdesmobility/secrets.env}}"

if [[ -f "${LOCAL_SECRETS_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCAL_SECRETS_ENV}"
  set +a
fi

setup_vasp_runtime_env() {
  local nvhpc_root="${NVHPC_ROOT:-}"
  local nv_arch="${NV_ARCH:-Linux_x86_64}"
  local nv_version="${NV_VERSION:-}"

  if [[ -n "${nvhpc_root}" ]]; then
    if [[ -z "${nv_version}" ]]; then
      nv_version="$(basename "$(ls -d "${nvhpc_root}/${nv_arch}"/* 2>/dev/null | sort -V | tail -n 1)")"
    fi
    local nv_base="${nvhpc_root}/${nv_arch}/${nv_version}"
    export NVHPC_ROOT="${nvhpc_root}"
    export NV_ARCH="${nv_arch}"
    export NV_VERSION="${nv_version}"
    export PATH="${nv_base}/comm_libs/mpi/bin:${nv_base}/compilers/bin:${PATH}"
    export LD_LIBRARY_PATH="${nv_base}/compilers/extras/qd/lib:${nv_base}/comm_libs/mpi/lib:${nv_base}/compilers/lib:${nv_base}/math_libs/lib64:${nv_base}/cuda/lib64:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "${VASP_BIN_DIR:-}" ]]; then
    export PATH="${VASP_BIN_DIR}:${PATH}"
  fi
}

setup_vasp_runtime_env

ROUND_INDEX="${1:-${ROUND_INDEX:-1}}"
FEEDBACK_BATCH_ROOT="${2:-${FEEDBACK_BATCH_ROOT:-}}"
if [[ -z "${FEEDBACK_BATCH_ROOT}" ]]; then
  echo "Usage: bash ${0##*/} <round_index> <completed_2d_mobility_batch_root>" >&2
  exit 2
fi

if ! [[ "${ROUND_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ROUND_INDEX must be an integer, got: ${ROUND_INDEX}" >&2
  exit 2
fi

printf -v ROUND_ID 'loop_%02d' "${ROUND_INDEX}"
ROUND_ROOT="${ROOT_DIR}/runs/${ROUND_ID}__closed_loop_round"
CONTROL_DIR="${ROUND_ROOT}/_control"
SESSION_NAME="${SESSION_NAME:-invdes-${ROUND_ID}}"
RUNNER_SESSION_NAME="${RUNNER_SESSION_NAME:-${SESSION_NAME}-runner}"
RUNNER_LOG="${CONTROL_DIR}/closed_loop_runner.log"
TMUX_CONTEXT="${CONTROL_DIR}/tmux_context.json"

MONGO_DB="${MONGO_DB:-materials_database}"
MONGO_COLLECTION="${MONGO_COLLECTION:-invdesmobility}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-100000}"
SAMPLES_PER_JOB="${SAMPLES_PER_JOB:-1000}"
TOP_K="${TOP_K:-10}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PHONONBENCH_GPU_LIST="${PHONONBENCH_GPU_LIST:-${GPU_LIST}}"

if [[ -e "${ROUND_ROOT}" ]]; then
  if [[ ! -d "${ROUND_ROOT}" ]]; then
    echo "round root exists but is not a directory: ${ROUND_ROOT}" >&2
    exit 2
  fi
  shopt -s nullglob dotglob
  existing_items=("${ROUND_ROOT}"/*)
  shopt -u nullglob dotglob
  if (( ${#existing_items[@]} > 0 )); then
    disallowed=()
    for item in "${existing_items[@]}"; do
      if [[ "$(basename "${item}")" != "_control" ]]; then
        disallowed+=("$(basename "${item}")")
      fi
    done
    if (( ${#disallowed[@]} > 0 )); then
      echo "round root already exists with run contents: ${ROUND_ROOT}" >&2
      echo "Choose a new ROUND_INDEX or remove the old round root first." >&2
      exit 2
    fi
  fi
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 2
fi

if tmux has-session -t "${RUNNER_SESSION_NAME}" 2>/dev/null; then
  echo "tmux runner session already exists: ${RUNNER_SESSION_NAME}" >&2
  echo "Please close it or override RUNNER_SESSION_NAME." >&2
  exit 2
fi

mkdir -p "${CONTROL_DIR}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

payload = {
    "round_index": int(${ROUND_INDEX@Q}),
    "round_id": ${ROUND_ID@Q},
    "feedback_batch_root": ${FEEDBACK_BATCH_ROOT@Q},
    "mongo_db": ${MONGO_DB@Q},
    "mongo_collection": ${MONGO_COLLECTION@Q},
    "total_samples": int(${TOTAL_SAMPLES@Q}),
    "samples_per_job": int(${SAMPLES_PER_JOB@Q}),
    "top_k": int(${TOP_K@Q}),
    "gpu_list": ${GPU_LIST@Q},
    "phononbench_gpu_list": ${PHONONBENCH_GPU_LIST@Q},
    "session_name": ${SESSION_NAME@Q},
    "runner_session_name": ${RUNNER_SESSION_NAME@Q},
    "runner_log": ${RUNNER_LOG@Q},
}
Path(${TMUX_CONTEXT@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

cmd=(
  env
  PYTHONUNBUFFERED=1
  "${PYTHON_BIN}" -m invdesmobility closed-loop-round
  --round-index "${ROUND_INDEX}"
  --feedback-batch-root "${FEEDBACK_BATCH_ROOT}"
  --mongo-db "${MONGO_DB}"
  --mongo-collection "${MONGO_COLLECTION}"
  --total-samples "${TOTAL_SAMPLES}"
  --samples-per-job "${SAMPLES_PER_JOB}"
  --top-k "${TOP_K}"
  --gpu-list "${GPU_LIST}"
  --phononbench-gpu-list "${PHONONBENCH_GPU_LIST}"
)

printf -v cmd_str '%q ' "${cmd[@]}"
printf -v runner_log_q '%q' "${RUNNER_LOG}"

RUNNER_SCRIPT="${CONTROL_DIR}/closed_loop_runner.sh"
cat > "${RUNNER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
setup_vasp_runtime_env() {
  local nvhpc_root="\${NVHPC_ROOT:-}"
  local nv_arch="\${NV_ARCH:-Linux_x86_64}"
  local nv_version="\${NV_VERSION:-}"

  if [[ -n "\${nvhpc_root}" ]]; then
    if [[ -z "\${nv_version}" ]]; then
      nv_version="\$(basename "\$(ls -d "\${nvhpc_root}/\${nv_arch}"/* 2>/dev/null | sort -V | tail -n 1)")"
    fi
    local nv_base="\${nvhpc_root}/\${nv_arch}/\${nv_version}"
    export NVHPC_ROOT="\${nvhpc_root}"
    export NV_ARCH="\${nv_arch}"
    export NV_VERSION="\${nv_version}"
    export PATH="\${nv_base}/comm_libs/mpi/bin:\${nv_base}/compilers/bin:\${PATH}"
    export LD_LIBRARY_PATH="\${nv_base}/compilers/extras/qd/lib:\${nv_base}/comm_libs/mpi/lib:\${nv_base}/compilers/lib:\${nv_base}/math_libs/lib64:\${nv_base}/cuda/lib64:\${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "\${VASP_BIN_DIR:-}" ]]; then
    export PATH="\${VASP_BIN_DIR}:\${PATH}"
  fi
}

setup_vasp_runtime_env
cd ${ROOT_DIR@Q}
${cmd_str} 2>&1 | tee ${runner_log_q}
status=\${PIPESTATUS[0]}
echo
echo "[runner] closed-loop command exited with status=\${status}. Session kept open for inspection."
exec bash -i
EOF
chmod +x "${RUNNER_SCRIPT}"

tmux new-session -d -s "${RUNNER_SESSION_NAME}" -c "${ROOT_DIR}" "${RUNNER_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" -c "${ROOT_DIR}" "bash -i"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
tmux send-keys -t "${SESSION_NAME}:0.0" "watch -n 30 \"bash ${ROOT_DIR}/scripts/show_closed_loop_round_progress.sh ${ROUND_ID}\"" C-m
tmux split-window -t "${SESSION_NAME}:0" -v -c "${ROOT_DIR}" "bash -i"
tmux send-keys -t "${SESSION_NAME}:0.1" "tail -n 80 -F ${RUNNER_LOG}" C-m
tmux select-layout -t "${SESSION_NAME}:0" even-vertical

echo "Started tmux session: ${SESSION_NAME}"
echo "runner session: ${RUNNER_SESSION_NAME}"
echo "round_id: ${ROUND_ID}"
echo "feedback_batch_root: ${FEEDBACK_BATCH_ROOT}"
echo "mongo target: ${MONGO_DB}.${MONGO_COLLECTION}"
echo "runner_log: ${RUNNER_LOG}"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "runner attach (avoid Ctrl+C unless you mean to interrupt the run): tmux attach -t ${RUNNER_SESSION_NAME}"
echo "progress: bash ${ROOT_DIR}/scripts/show_closed_loop_round_progress.sh ${ROUND_ID}"
echo "watch: watch -n 30 'bash ${ROOT_DIR}/scripts/show_closed_loop_round_progress.sh ${ROUND_ID}'"
