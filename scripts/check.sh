#!/usr/bin/env bash
# What CI runs, locally: lint, format check, tests. Run before every push.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff check .   # the ruff pinned in .[dev], not whatever is on PATH
python -m ruff format --check .
python -m pytest -q
