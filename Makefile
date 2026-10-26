.PHONY: help install run dev check-system worker-install worker-dev worker-deploy clean

PYTHON ?= python3
VENV   ?= .venv

help:
	@echo "Targets:"
	@echo "  make install         — Create a Python venv and install deps (via uv)"
	@echo "  make run             — Run the app"
	@echo "  make dev             — Run the app in dev mode (DEBUG logging)"
	@echo "  make check-system    — Check Linux system dependencies"
	@echo "  make worker-install  — Install Cloudflare Worker deps (npm)"
	@echo "  make worker-dev      — Run the Worker locally on port 8787"
	@echo "  make worker-deploy   — Deploy the Worker to Cloudflare"
	@echo "  make clean           — Remove build artifacts"

install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	uv venv $(VENV)
	uv pip install --python $(VENV)/bin/python -e ".[dev]"
	@echo ""
	@echo "Install complete. To run: source $(VENV)/bin/activate && python -m clicky"

run:
	$(VENV)/bin/python -m clicky

dev:
	LOG_LEVEL=DEBUG $(VENV)/bin/python -m clicky

check-system:
	@bash scripts/check-system.sh

worker-install:
	cd worker && npm install

worker-dev:
	cd worker && npx wrangler dev

worker-deploy:
	cd worker && npx wrangler deploy

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
