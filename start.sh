#!/usr/bin/env bash
set -euo pipefail

# Override these at launch when needed, for example:
# MODEL_PATH=/data/ShizhenGPT PORT=7861 ./start.sh
export MODEL_ID="${MODEL_ID:-FreedomIntelligence/ShizhenGPT-7B-LLM}"
export MODEL_PATH="${MODEL_PATH:-/home/featurize/data/models/ShizhenGPT-7B-LLM}"
export PORT="${PORT:-7860}"
export SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
export HF_HOME="${HF_HOME:-/home/featurize/data/hf-cache}"
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

# On Featurize, prefer its preinstalled CUDA-enabled base interpreter. On
# other systems, use an explicitly supplied interpreter or the PATH default.
if [[ -z "${PYTHON_BIN:-}" && -x /environment/miniconda3/bin/python ]]; then
  PYTHON_BIN=/environment/miniconda3/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

exec "$PYTHON_BIN" app.py
