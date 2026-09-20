# Shortcuts for the commands in the README. `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help install test lint format check backfill features train predict serve ui docker-up docker-down clean

BACKFILL_START ?= 2025-09-01
BACKFILL_END   ?= 2026-09-01

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies from the lockfile
	uv sync --all-extras

test:  ## Run the unit tests
	uv run pytest

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Reformat the code
	uv run ruff format .
	uv run ruff check --fix .

check: lint test  ## Everything CI runs

backfill:  ## Load history into the feature store (BACKFILL_START/BACKFILL_END)
	uv run padel backfill --start $(BACKFILL_START) --end $(BACKFILL_END)

features:  ## Run the feature pipeline
	uv run padel feature-pipeline

train:  ## Run the training pipeline
	uv run padel training-pipeline

predict:  ## Run the inference pipeline
	uv run padel inference-pipeline

serve:  ## Run the prediction API
	uv run padel serve

ui:  ## Run the Streamlit UI
	uv run streamlit run ui/app.py

docker-up:  ## Start mlflow + api + ui
	docker compose up --build

docker-down:  ## Stop the stack
	docker compose down

clean:  ## Remove local data, models and caches
	rm -rf data mlruns mlartifacts mlflow.db .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
