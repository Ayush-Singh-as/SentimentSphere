# SentimentSphere — developer entry points.
#
# Everything a contributor or CI needs is a target here. If a command is worth
# running twice, it belongs in this file rather than in a README code block.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Determinism: PYTHONHASHSEED only takes effect if set before the interpreter
# starts, so it is exported here rather than set inside Python.
export PYTHONHASHSEED := 1337

UV := uv
PY := $(UV) run

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
.PHONY: install
install:  ## Install core + dev deps and the package (editable)
	$(UV) sync --extra dev
	$(UV) run pre-commit install
	@echo "-> try: make test && $(UV) run sphere labels"

.PHONY: install-all
install-all:  ## Install every extra (torch, transformers, cv, serving, demo) — large
	$(UV) sync --all-extras

.PHONY: install-legacy
install-legacy:  ## Add TensorFlow, needed only to score the v1 Keras artifacts
	$(UV) sync --extra dev --extra legacy

.PHONY: lock
lock:  ## Refresh uv.lock
	$(UV) lock

# --------------------------------------------------------------------------- #
# Quality gates — the same commands CI runs
# --------------------------------------------------------------------------- #
.PHONY: fmt
fmt:  ## Auto-format and auto-fix
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

.PHONY: lint
lint:  ## Lint (no writes)
	$(PY) ruff check src tests
	$(PY) ruff format --check src tests

.PHONY: typecheck
typecheck:  ## mypy, strict on src/
	$(PY) mypy src

.PHONY: test
test:  ## Run tests that need no datasets or weights
	$(PY) pytest -m "not needs_data and not needs_artifacts and not slow and not gpu"

.PHONY: test-all
test-all:  ## Run every test, including dataset- and weight-dependent ones
	$(PY) pytest

.PHONY: cov
cov:  ## Tests with coverage report
	$(PY) pytest --cov --cov-report=term-missing --cov-report=xml \
	  -m "not needs_data and not needs_artifacts and not gpu"

.PHONY: check
check: lint typecheck test  ## Everything CI enforces

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
.PHONY: data
data:  ## Download + checksum every dataset (Phase 1)
	$(PY) sphere data

.PHONY: data-status
data-status:  ## Report which datasets are present locally
	@echo "text aggregate : $$([ -f data/raw/text_emotion_aggregate.csv ] && echo present || echo MISSING)"
	@echo "TESS wavs      : $$(find data/raw/tess -name '*.wav' 2>/dev/null | wc -l) / 2800"
	@echo "RAVDESS wavs   : $$(find data/raw/ravdess -name '*.wav' 2>/dev/null | wc -l)"
	@echo "CREMA-D wavs   : $$(find data/raw/crema_d -name '*.wav' 2>/dev/null | wc -l)"
	@echo "SAVEE wavs     : $$(find data/raw/savee -name '*.wav' 2>/dev/null | wc -l)"
	@echo "FER-2013 imgs  : $$(find data/raw/fer2013 -name '*.jpg' 2>/dev/null | wc -l)"
	@echo "MELD           : $$([ -d data/raw/meld ] && echo present || echo MISSING)"
	@echo "v1 artifacts   : $$(ls artifacts/v1 2>/dev/null | wc -l) files"

# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
.PHONY: eval-baseline
eval-baseline:  ## Score every v1 artifact through the v2 harness (Phase 1)
	$(PY) sphere eval --baseline --all

.PHONY: train-text train-audio train-vision train-fusion
train-text:  ## Fine-tune the text head
	$(PY) sphere train --config configs/text/deberta_v3_base.yaml
train-audio:  ## Fine-tune the speech head
	$(PY) sphere train --config configs/audio/wavlm_large.yaml
train-vision:  ## Fine-tune the face head
	$(PY) sphere train --config configs/vision/convnext_tiny.yaml
train-fusion:  ## Fit the fusion meta-classifier
	$(PY) sphere train --config configs/fusion/meta_lr.yaml

.PHONY: reports
reports:  ## Regenerate every figure and table in reports/
	$(PY) sphere eval --all --write-reports

# --------------------------------------------------------------------------- #
# Serve / demo
# --------------------------------------------------------------------------- #
.PHONY: serve
serve:  ## Run the FastAPI inference server on :8000
	$(PY) uvicorn sentimentsphere.serving.api:app --reload --port 8000

.PHONY: demo
demo:  ## Run the Gradio app locally (same code as the HF Space)
	$(PY) python apps/space/app.py

.PHONY: docker
docker:  ## Build the Space container
	docker build -t sentimentsphere:local -f apps/space/Dockerfile .

.PHONY: docker-run
docker-run: docker  ## Build and run the Space container on :7860
	docker run --rm -p 7860:7860 sentimentsphere:local

# --------------------------------------------------------------------------- #
# Docs
# --------------------------------------------------------------------------- #
.PHONY: docs docs-serve
docs:  ## Build the docs site
	$(PY) mkdocs build --strict
docs-serve:  ## Serve docs with live reload on :8001
	$(PY) mkdocs serve -a localhost:8001

# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #
.PHONY: clean
clean:  ## Remove caches and build output (leaves data/ and artifacts/ alone)
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true

.PHONY: repo-size
repo-size:  ## Show git and working-tree size (regression check on repo hygiene)
	@echo ".git         : $$(du -sh .git | cut -f1)"
	@echo "working tree : $$(du -sh --exclude=.git . | cut -f1)"
	@echo "largest tracked blobs:"
	@git ls-tree -r -l HEAD | sort -k4 -rn | head -5 | awk '{printf "  %10d  %s\n", $$4, $$5}'
