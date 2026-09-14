#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade-strategy only-if-needed -r requirements.txt

echo "Dependencies installed. Download the model, then run ./start.sh."
