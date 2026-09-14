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

# Prefer the repository virtual environment on a fresh deployment, while
# retaining compatibility with the Featurize base environment used originally.
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
fi

exec "$PYTHON_BIN" app.py
