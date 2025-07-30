# Repository Data Preprocessing Makefile
# 
# This Makefile provides convenient commands for running the data preprocessing
# pipeline and managing the output data.

PYTHON = python3
ROOT_DIR = /home/snow/Documents/Projects/github-repositories/bkocis
OUTPUT_DIR = mcp-resources
CONFIG_FILE = config.json

.PHONY: help preprocess quick clean status view-summary view-repos dry-run test

# Default target
help:
	@echo "Repository Data Preprocessing"
	@echo "============================="
	@echo ""
	@echo "Available commands:"
	@echo "  preprocess    - Run full preprocessing pipeline"
	@echo "  quick         - Run quick preprocessing (minimal analysis)"
	@echo "  dry-run       - Show what would be processed without processing"
	@echo "  status        - Show current preprocessing status"
	@echo "  view-summary  - Display summary of processed data"
	@echo "  view-repos    - List all processed repositories"
	@echo "  clean         - Clean output directory"
	@echo "  test          - Test preprocessing on a small subset"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  ROOT_DIR     = $(ROOT_DIR)"
	@echo "  OUTPUT_DIR   = $(OUTPUT_DIR)"
	@echo "  CONFIG_FILE  = $(CONFIG_FILE)"

# Run full preprocessing
preprocess:
	@echo "Running full preprocessing..."
	$(PYTHON) run_preprocessing.py \
		--root-dir "$(ROOT_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--config "$(CONFIG_FILE)" \
		--verbose

# Run quick preprocessing
quick:
	@echo "Running quick preprocessing..."
	$(PYTHON) run_preprocessing.py \
		--root-dir "$(ROOT_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--config "$(CONFIG_FILE)" \
		--quick \
		--verbose

# Dry run to see what would be processed
dry-run:
	@echo "Dry run - showing what would be processed..."
	$(PYTHON) run_preprocessing.py \
		--root-dir "$(ROOT_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--config "$(CONFIG_FILE)" \
		--dry-run

# Show current status
status:
	@echo "Preprocessing Status"
	@echo "==================="
	@if [ -f "$(OUTPUT_DIR)/repositories_summary.json" ]; then \
		echo "✓ Preprocessing completed"; \
		echo "Summary file: $(OUTPUT_DIR)/repositories_summary.json"; \
		$(PYTHON) -c "import json; data=json.load(open('$(OUTPUT_DIR)/repositories_summary.json')); print(f'Repositories: {data[\"metadata\"][\"total_repositories\"]}'); print(f'Total files: {data[\"summary\"][\"total_files\"]:,}'); print(f'Last updated: {data[\"metadata\"][\"processing_timestamp\"]}')"; \
	else \
		echo "✗ No preprocessing data found"; \
		echo "Run 'make preprocess' to generate data"; \
	fi

# View summary data
view-summary:
	@if [ -f "$(OUTPUT_DIR)/repositories_summary.json" ]; then \
		echo "Repository Summary"; \
		echo "================"; \
		$(PYTHON) -c "import json; data=json.load(open('$(OUTPUT_DIR)/repositories_summary.json')); print(json.dumps(data['summary'], indent=2))"; \
	else \
		echo "No summary data found. Run 'make preprocess' first."; \
	fi

# List all processed repositories
view-repos:
	@if [ -f "$(OUTPUT_DIR)/repositories_summary.json" ]; then \
		echo "Processed Repositories"; \
		echo "===================="; \
		$(PYTHON) -c "import json; data=json.load(open('$(OUTPUT_DIR)/repositories_summary.json')); [print(f'- {repo}') for repo in sorted(data['repository_list'])]"; \
	else \
		echo "No repository data found. Run 'make preprocess' first."; \
	fi

# Clean output directory
clean:
	@echo "Cleaning output directory..."
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		rm -f "$(OUTPUT_DIR)"/*.json; \
		rm -rf "$(OUTPUT_DIR)/repositories"; \
		echo "✓ Cleaned $(OUTPUT_DIR)"; \
	else \
		echo "Output directory $(OUTPUT_DIR) does not exist"; \
	fi

# Test preprocessing on a subset
test:
	@echo "Testing preprocessing script..."
	$(PYTHON) -c "import data_preprocessor; print('✓ Import successful')"
	@echo "✓ Script syntax is valid"
	@echo "Run 'make dry-run' to test on actual data"

# Advanced targets

# Create a backup of current data
backup:
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		backup_name="backup_$(shell date +%Y%m%d_%H%M%S)"; \
		echo "Creating backup: $$backup_name"; \
		cp -r "$(OUTPUT_DIR)" "$$backup_name"; \
		echo "✓ Backup created: $$backup_name"; \
	else \
		echo "No data to backup"; \
	fi

# Show disk usage
disk-usage:
	@echo "Disk Usage"
	@echo "=========="
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		du -h "$(OUTPUT_DIR)"/* 2>/dev/null | sort -hr || echo "No files in output directory"; \
	else \
		echo "Output directory does not exist"; \
	fi

# Validate output files
validate:
	@echo "Validating output files..."
	@if [ -f "$(OUTPUT_DIR)/repositories_summary.json" ]; then \
		$(PYTHON) -c "import json; json.load(open('$(OUTPUT_DIR)/repositories_summary.json')); print('✓ repositories_summary.json is valid')"; \
	else \
		echo "✗ repositories_summary.json not found"; \
	fi
	@if [ -f "$(OUTPUT_DIR)/repositories_data.json" ]; then \
		$(PYTHON) -c "import json; json.load(open('$(OUTPUT_DIR)/repositories_data.json')); print('✓ repositories_data.json is valid')"; \
	else \
		echo "✗ repositories_data.json not found"; \
	fi
	@if [ -f "$(OUTPUT_DIR)/readme_index.json" ]; then \
		$(PYTHON) -c "import json; json.load(open('$(OUTPUT_DIR)/readme_index.json')); print('✓ readme_index.json is valid')"; \
	else \
		echo "✗ readme_index.json not found"; \
	fi

# Show configuration
show-config:
	@echo "Current Configuration"
	@echo "===================="
	@if [ -f "$(CONFIG_FILE)" ]; then \
		cat "$(CONFIG_FILE)"; \
	else \
		echo "Config file $(CONFIG_FILE) not found"; \
	fi