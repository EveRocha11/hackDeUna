UV ?= uv

.PHONY: setup dataset-build dataset-validate evals-build format lint type-check

# -----------------------------
# Environment Setup
# -----------------------------

setup:
	$(UV) sync --extra dev

# -----------------------------
# Data & Evaluation Artifacts
# -----------------------------

dataset-build:
	# Generate synthetic CSV + DuckDB dataset artifacts.
	$(UV) run dataset-generate

dataset-validate:
	# Validate generated dataset integrity and consistency checks.
	$(UV) run dataset-validate

evals-build:
	# Build deterministic evaluation set JSON from DuckDB facts.
	$(UV) run evals-build

# -----------------------------
# Code Quality
# -----------------------------

format:
	# Auto-format code.
	$(UV) run ruff format src tests

lint:
	# Run static lint checks.
	$(UV) run ruff check src tests

type-check:
	# Run static type checking.
	$(UV) run mypy src
