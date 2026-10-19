# Frontend Changes

## Code quality (black)

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

## API tests

No front-end files changed. This feature is backend test infrastructure, so the "front-end only" scope in the request does not apply; it was built as specified.

## Changes

- `backend/tests/conftest.py`: shared fixtures (`mock_rag_system`, `sample_sources`, `sample_courses`, `client`) and `create_test_app`, which mirrors the `/api/query`, `/api/courses` and `/` routes without the static-file mount or real `RAGSystem`.
- `backend/tests/test_api.py`: 16 tests covering success, session handling, validation (422), server errors (500), wrong methods (405) and unknown routes (404).
- `pyproject.toml`: `[tool.pytest.ini_options]` (`testpaths`, `pythonpath`, `addopts`) and a `dev` dependency group (`pytest`, `httpx`).

## Run

`uv run --group dev pytest`

## Note

The test app duplicates the route logic from `backend/app.py`, so route changes there must be mirrored in `conftest.py`. The `/` test checks a stub response, not the real static frontend.

## Theme toggle


## index.html
- Added a fixed top-right icon button (`#themeToggle`) with sun and moon SVGs. It is a native `<button>`, so Tab/Enter/Space work, and `aria-label` updates with the current theme.
- Added an inline head script that reads `localStorage.theme` and sets `data-theme` on `<html>` before first paint (no flash).
- Bumped cache-busting versions to `?v=10`.

## style.css
- Added `:root[data-theme="light"]` overriding all color variables (light background and surfaces, dark text `#0f172a`/`#475569`, `#cbd5e1` borders, same blue primary). Dark remains the default.
- Added new variables (`--code-bg`, `--error-text`, `--success-text`) and replaced hardcoded colors for code blocks, error/success messages and the welcome shadow so they work in both themes. Error/success text is darker in light mode for contrast.
- Added `.theme-toggle` styles: circular button, hover scale, `:focus-visible` outline, and a rotate/fade crossfade between sun (dark theme) and moon (light theme).
- Added a `.theme-transition` class that applies a 0.3s color/border/shadow transition to all elements, only while toggling, so normal interactions are unaffected.

## script.js
- Added `setupThemeToggle()`: toggles `data-theme` on `<html>` between `dark` and `light`, persists the choice in `localStorage`, and updates the button's `aria-label`.
