UV ?= uv

.PHONY: setup dataset-build dataset-validate format lint type-check

setup:
	$(UV) sync --extra dev

dataset-build:
	$(UV) run dataset-generate

dataset-validate:
	$(UV) run dataset-validate

format:
	$(UV) run ruff format src tests

lint:
	$(UV) run ruff check src tests

type-check:
	$(UV) run mypy src
