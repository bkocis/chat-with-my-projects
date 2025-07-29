# Credit Card MCP Server Docker Management Makefile

# Configuration
IMAGE_NAME := credit-card-mcp
CONTAINER_NAME := credit-card-mcp-server

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Default target
.DEFAULT_GOAL := help

# Phony targets (don't create files)
.PHONY: help build run stop restart logs shell cleanup status

help: ## Show this help message
	@echo "Credit Card MCP Server Docker Management"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-10s - %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker image
	@echo -e "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo -e "$(GREEN)Build completed!$(NC)"

run: ## Run the container with docker-compose
	@echo -e "$(BLUE)Starting container with docker-compose...$(NC)"
	docker-compose up -d
	@echo -e "$(GREEN)Container started!$(NC)"
	@echo -e "$(YELLOW)Use 'make logs' to see the output$(NC)"

stop: ## Stop the running container
	@echo -e "$(BLUE)Stopping container...$(NC)"
	docker-compose down
	@echo -e "$(GREEN)Container stopped!$(NC)"

restart: ## Restart the container
	@echo -e "$(BLUE)Restarting container...$(NC)"
	docker-compose restart
	@echo -e "$(GREEN)Container restarted!$(NC)"

logs: ## Show container logs
	@echo -e "$(BLUE)Showing container logs...$(NC)"
	docker-compose logs -f

shell: ## Open shell in running container
	@echo -e "$(BLUE)Opening shell in container...$(NC)"
	docker-compose exec credit-card-mcp /bin/bash

cleanup: ## Remove container and image
	@echo -e "$(BLUE)Cleaning up containers and images...$(NC)"
	docker-compose down --remove-orphans
	-docker rmi $(IMAGE_NAME) 2>/dev/null
	@echo -e "$(GREEN)Cleanup completed!$(NC)"

status: ## Show container status
	@echo -e "$(BLUE)Container status:$(NC)"
	docker-compose ps
	@echo ""
	@echo -e "$(BLUE)Resource usage:$(NC)"
	-docker stats $(CONTAINER_NAME) --no-stream 2>/dev/null || echo "Container not running" 

inspector: 
	npx @modelcontextprotocol/inspector

build_run_clean: ## Build image, run container, and clean up previous resources
	@echo -e "$(BLUE)Starting build, run, and clean process...$(NC)"
	@echo -e "$(BLUE)Step 1: Cleaning up any existing resources...$(NC)"
	docker-compose down --remove-orphans 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null
	@echo -e "$(GREEN)Cleanup completed!$(NC)"
	@echo ""
	@echo -e "$(BLUE)Step 2: Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo -e "$(GREEN)Build completed!$(NC)"
	@echo ""
	@echo -e "$(BLUE)Step 3: Starting container with docker-compose...$(NC)"
	docker-compose up -d
	@echo -e "$(GREEN)Container started!$(NC)"
	@echo -e "$(YELLOW)Use 'make logs' to see the output$(NC)"
	@echo -e "$(GREEN)Build, run, and clean process completed successfully!$(NC)"

prune:
	docker compose down ; docker rmi -f chat-with-my-github-github-repos-mcp:latest chat-with-my-github-github-repos-client:latest ; docker system prune -f ; docker compose up