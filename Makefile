# Reproducible, CPU-only entry points.
#
# Every target here runs without a GPU, a container, or a model download.
# `make validate` is the single command a reviewer should run.

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

# Fall back to the system interpreter when the venv has not been created yet,
# so `make help` and `make test` still work on a fresh checkout.
ifeq (,$(wildcard $(PYTHON)))
PYTHON := python3
PIP    := pip3
endif

.DEFAULT_GOAL := help
.PHONY: help venv install test validate validate-full lint audit reports sheets \
        download generate build clean-check review review-summary

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

venv: ## Create the virtualenv
	uv venv .venv --python 3.11

install: ## Install the package with data and test extras (CPU only)
	$(PIP) install -e ".[data,test]"

# ---------------------------------------------------------------------------
# Checks -- these are the CPU-only gates
# ---------------------------------------------------------------------------

test: ## Run the unit tests
	$(PYTHON) -m pytest tests/ -q

validate: ## Validate the prepared dataset (structural + TAO schema)
	$(PYTHON) -m temporal_mcq validate --skip-decode

validate-full: ## Validate including video decode and frame-count probing
	$(PYTHON) -m temporal_mcq validate --max-frame-probes 0

check: test validate ## Run tests and dataset validation

# ---------------------------------------------------------------------------
# Pipeline stages (require the downloaded source dataset)
# ---------------------------------------------------------------------------

download: ## Download the source dataset into data/raw/bridgedata2/
	$(PYTHON) -m temporal_mcq download

audit: ## Write reports/dataset_audit.md from the source dataset
	$(PYTHON) -m temporal_mcq audit

generate: ## Generate candidate MCQ questions from structured captions
	$(PYTHON) -m temporal_mcq generate

build: ## Split by source video and emit TAO annotations
	$(PYTHON) -m temporal_mcq build

reports: ## Write the dataset summary, quality and split reports
	$(PYTHON) -m temporal_mcq reports

sheets: ## Build contact sheets for human visual review (writes JPEGs)
	$(PYTHON) -m temporal_mcq sheets

# ---------------------------------------------------------------------------
# Human visual verification (local only; nothing is uploaded)
# ---------------------------------------------------------------------------

review: ## Serve the local review interface at http://127.0.0.1:8765/
	$(PYTHON) -m temporal_mcq review-server --open

review-summary: ## Print the review status summary
	$(PYTHON) -m temporal_mcq review-summary

# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

clean-check: ## Report files that would be committed, and flag anything large
	@git status --short
	@echo "--- files staged for commit larger than 1 MB (should be none) ---"
	@git ls-files -z 2>/dev/null | xargs -0 -r du -h 2>/dev/null \
	  | awk '$$1 ~ /M|G/ {print}' || true
