# GLM-5.2-AWQ-INT4 vLLM deployment. All config lives in .env.
# Run `make` or `make help` to list targets.

.DEFAULT_GOAL := help

COMPOSE := docker compose
PORT    := $(shell sed -n 's/^HOST_PORT=//p' .env 2>/dev/null)
PORT    := $(if $(PORT),$(PORT),8000)

.PHONY: help preflight pull start stop restart logs ps config health smoke bench

help: ## Show this help
	@awk 'BEGIN {FS=":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

preflight: ## Read-only env + GPU/model checks (run before start)
	@./scripts/preflight.sh

pull: ## Pull the vLLM image (~6 GB)
	@$(COMPOSE) pull

start: ## Start the server detached (loads 440 GB, ~6 min)
	@$(COMPOSE) up -d

stop: ## Stop the server (keeps the container)
	@$(COMPOSE) stop

restart: stop start ## Restart (stop then start)

logs: ## Tail server logs (Ctrl-C to detach)
	@$(COMPOSE) logs -f vllm

ps: ## Show container status
	@$(COMPOSE) ps

config: ## Render effective compose config (flags resolved from .env)
	@$(COMPOSE) config

health: ## Probe the /health endpoint
	@curl -sf http://localhost:$(PORT)/health >/dev/null && echo "healthy (port $(PORT))" || (echo "DOWN (port $(PORT))"; exit 1)

smoke: ## One-shot chat completion (reasoning model)
	@curl -s http://localhost:$(PORT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"glm-5.2-awq-int4","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":1024}' \
	  | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]; m=c["message"]; print(m["content"] or ("[finish=%s] " % c["finish_reason"]) + (m.get("reasoning") or "")[:200])'

bench: ## Measure prefill + decode TPS (server must be running)
	@python3 scripts/bench.py
