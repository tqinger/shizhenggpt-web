#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" && -x /environment/miniconda3/bin/python ]]; then
  PYTHON_BIN=/environment/miniconda3/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

"$PYTHON_BIN" -m pip install --user --upgrade-strategy only-if-needed -r requirements.txt

echo "Dependencies installed. Download the model, then run ./start.sh."
