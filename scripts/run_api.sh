#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
