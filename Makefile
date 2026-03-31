.PHONY: help install dev test clean run format lint setup serve-api serve-api-prod \
        docker-build docker-up docker-up-detached docker-down docker-logs setup-codespaces

help: ## Show this help
	@echo "NotionHealth AI — Available Commands"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────

setup: ## Create venv and install core deps
	@echo "Setting up NotionHealth AI..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Next steps:"
	@echo "  1. cp .env.example .env  — fill in your API keys"
	@echo "  2. source venv/bin/activate"
	@echo "  3. make setup-notion     — create Notion databases"
	@echo "  4. make serve-api        — start web GUI at http://localhost:8000"

setup-notion: ## Create/seed Notion databases and dashboard
	python scripts/setup_notion.py

install: ## Install dependencies into active venv
	pip install -r requirements.txt

dev: ## Install dev dependencies into active venv
	pip install -e ".[dev]"

# ── Web server ─────────────────────────────────────────────────────────────────

serve-api: ## Start web GUI + REST API on http://localhost:8000 (auto-reload)
	uvicorn notion_health_ai.api:app --reload --host 127.0.0.1 --port 8000

serve-api-prod: ## Start web GUI + REST API (production, no reload)
	uvicorn notion_health_ai.api:app --host 0.0.0.0 --port 8000 --workers 2

# ── Tests ──────────────────────────────────────────────────────────────────────

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-verbose: ## Run tests with verbose output and HTML coverage
	pytest tests/ -v -s --cov=src --cov-report=html

# ── Code quality ───────────────────────────────────────────────────────────────

format: ## Format code with black + isort
	black src/ tests/
	isort src/ tests/

lint: ## Run flake8 + mypy
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/

check: format lint test ## Run all checks (format → lint → test)

# ── CLI shortcuts ──────────────────────────────────────────────────────────────

run: ## Run the interactive CLI
	notion-health interactive

run-mcp: ## Run the MCP server (for Claude Desktop)
	notion-health-mcp

# ── Build / publish ────────────────────────────────────────────────────────────

build: ## Build distribution package
	python -m pip install --upgrade build
	python -m build

publish: ## Publish to PyPI (requires credentials)
	python -m pip install --upgrade twine
	python -m twine upload dist/*

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t notion-health-ai .

docker-up: ## Build and start full stack (API + web GUI)
	docker compose up --build

docker-up-detached: ## Start full stack in background
	docker compose up --build -d

docker-down: ## Stop docker compose stack
	docker compose down

docker-logs: ## Tail docker compose logs
	docker compose logs -f

# ── GitHub Codespaces ─────────────────────────────────────────────────────────

setup-codespaces: ## Install deps in Codespaces (called by devcontainer postCreateCommand)
	pip install --upgrade pip
	pip install -e ".[dev]"
	@echo "Run: make serve-api  to start the web GUI at http://localhost:8000"

# ── Clean ──────────────────────────────────────────────────────────────────────

clean: ## Remove build/test artefacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build htmlcov .coverage coverage.xml
