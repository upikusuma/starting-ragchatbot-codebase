# Frontend Changes

No front-end files (`frontend/index.html`, `script.js`, `style.css`) were changed.

This feature (code quality tooling with black) applies to Python only. Scope was confirmed with the user: Python tooling, no front-end formatter.

## What changed

- `pyproject.toml`: added `black` to a `dev` dependency group and a `[tool.black]` config (line length 88, py313, excludes `.venv`, `chroma_db`, `docs`).
- `scripts/format.sh`: runs `uv run black .` to format the codebase.
- `scripts/check.sh`: runs `uv run black --check --diff .` and exits non-zero if formatting is off (CI-friendly).
- Reformatted 9 files in `backend/` with black.

## Usage

```
uv sync --group dev
./scripts/format.sh
./scripts/check.sh
```
