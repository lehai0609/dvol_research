# Repository Guidelines

## Project Structure & Module Organization
- Root: Python research workspace (package mode disabled). Core config in `pyproject.toml`.
- Docs: `s01_problem_statement.md`, `s02_research_design_plan.md`, `s03_TRD.md` capture requirements and design.
- Tests: `tests/` with `test_*.py` (e.g., `tests/test_sanity.py`).
- Add code modules at the repo root or create folders like `research/`, `scripts/`, and `notebooks/` as needed. Keep datasets out of version control.

## Build, Test, and Development Commands
- `poetry install`: Install dependencies defined in `pyproject.toml`.
- `poetry shell`: Start a virtualenv shell for local commands.
- `poetry run pytest -q`: Run tests once, quiet output.
- `pytest -q`: Same as above when inside `poetry shell`.
- `poetry run python path/to/script.py`: Execute a script within the env.

## Coding Style & Naming Conventions
- Language: Python 3.11 (see `requires-python`). Use 4‑space indentation.
- Style: Follow PEP 8 and prefer type hints. Add docstrings for modules, classes, and public functions.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Logging: Prefer `loguru` for structured logs; use `rich` for readable CLI output when helpful.

## Testing Guidelines
- Framework: `pytest` (configured via `pyproject.toml`).
- Location: Place tests under `tests/` and name files `test_*.py`.
- Scope: Add unit tests for new logic; keep tests deterministic. Use fixtures over ad‑hoc setup.
- Run: `poetry run pytest -q` locally and before opening a PR.

## Commit & Pull Request Guidelines
- Commits: Use short, imperative subjects (e.g., "Add DVOL loader"). Keep changes focused.
- Messages: Include a brief body explaining what/why when nontrivial.
- PRs: Provide a clear description, link related issues, include screenshots/log excerpts when relevant, and list test coverage notes. Update docs (the `s0*` files or `README.md`) when behavior changes.

## Security & Configuration Tips
- Secrets: Store credentials in a local `.env` and load with `python-dotenv`. Do not commit `.env` or raw data.
- Config: Prefer YAML for experiment settings (`pyyaml` is available). Document defaults and paths in the PR.
