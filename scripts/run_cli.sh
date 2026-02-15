# scripts/run_cli.sh
#!/bin/bash

# Move to project root
cd "$(dirname "$0")/.."

export PYTHONPATH=.

python scripts/test_rag.py
