#!/usr/bin/env bash
# Run code quality checks (no changes made). Exits non-zero on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run black --check --diff .
