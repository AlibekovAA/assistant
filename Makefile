.PHONY: install fix check run

install:
	uv sync

fix:
	uv run ruff check --fix .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

run:
	uv run python -u -m assistant
