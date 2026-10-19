# Frontend Changes

No front-end files changed. This feature is backend test infrastructure, so the "front-end only" scope in the request does not apply; it was built as specified.

## Changes

- `backend/tests/conftest.py`: shared fixtures (`mock_rag_system`, `sample_sources`, `sample_courses`, `client`) and `create_test_app`, which mirrors the `/api/query`, `/api/courses` and `/` routes without the static-file mount or real `RAGSystem`.
- `backend/tests/test_api.py`: 16 tests covering success, session handling, validation (422), server errors (500), wrong methods (405) and unknown routes (404).
- `pyproject.toml`: `[tool.pytest.ini_options]` (`testpaths`, `pythonpath`, `addopts`) and a `dev` dependency group (`pytest`, `httpx`).

## Run

`uv run --group dev pytest`

## Note

The test app duplicates the route logic from `backend/app.py`, so route changes there must be mirrored in `conftest.py`. The `/` test checks a stub response, not the real static frontend.
