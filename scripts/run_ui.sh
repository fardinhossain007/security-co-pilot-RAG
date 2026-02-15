#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

streamlit run app/ui.py
