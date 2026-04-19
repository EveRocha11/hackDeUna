UV ?= uv

.PHONY: setup dataset-build dataset-validate evals-build evals-agent evals-agent-fast evals-agent-langsmith agent-dev agent-up agent-build api-dev format lint type-check

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

evals-agent:
	# Run full automatic agent evaluation (trajectory + quality).
	$(UV) run evals-agent

evals-agent-fast:
	# Run fast smoke evaluation over first 6 items.
	$(UV) run evals-agent --limit 6 --report data/evals/reports/agent_eval_report_fast.json

evals-agent-langsmith:
	# Run full baseline in LangSmith and create/update the evaluation dataset.
	$(UV) run evals-agent \
		--langsmith-dataset deuna-agent-eval-v1 \
		--langsmith-experiment-prefix deuna-agent-baseline \
		--langsmith-max-concurrency 5 \
		--langsmith-update-dataset \
		--report data/evals/reports/agent_eval_report_langsmith.json

agent-dev:
	# Run LangGraph agent server in local dev mode.
	$(UV) run langgraph dev

agent-up:
	# Run LangGraph agent server in Docker-based validation mode.
	$(UV) run langgraph up --watch

agent-build:
	# Build LangGraph agent server Docker image.
	$(UV) run langgraph build -t deuna-agent

api-dev:
	# Run FastAPI app for frontend integration testing.
	set -a; . ./.env; set +a; $(UV) run uvicorn server.api.app:app --host 0.0.0.0 --port 8000 --reload

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
