.PHONY: install format lint check

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

check:
	uv run ruff check .
	uv run ruff format --check .
