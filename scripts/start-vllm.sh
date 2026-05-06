#!/bin/bash
# Auto-compute --gpu-memory-utilization from the host GPU's total memory.
#
# Usage (in docker compose):
#   entrypoint: ["/scripts/start-vllm.sh"]
#   command: ["<headroom_gb>", <vllm_args...>]
#
# headroom_gb is the absolute GB the model needs (weights + KV cache + slack).
# The script converts that to a fraction of the detected GPU's total memory.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "[start-vllm] error: missing headroom GB as first arg" >&2
  exit 1
fi

HEADROOM_GB="$1"
shift

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[start-vllm] error: nvidia-smi not found inside container" >&2
  exit 1
fi

GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
if [[ -z "${GPU_MB}" ]]; then
  echo "[start-vllm] error: failed to read GPU memory" >&2
  exit 1
fi

UTIL=$(awk -v hb="${HEADROOM_GB}" -v gm="${GPU_MB}" '
  BEGIN {
    gg = gm / 1024.0
    if (gg <= 0) { print "0.95"; exit }
    v = hb / gg
    if (v > 0.95) v = 0.95
    if (v < 0.04) v = 0.04
    printf "%.3f", v
  }
')

echo "[start-vllm] GPU=$(awk -v gm=${GPU_MB} 'BEGIN{printf "%.1f", gm/1024}')G headroom=${HEADROOM_GB}G → --gpu-memory-utilization=${UTIL}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --gpu-memory-utilization "${UTIL}" \
  "$@"
