#!/usr/bin/env bash
# Format all Python code with black.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run black .
