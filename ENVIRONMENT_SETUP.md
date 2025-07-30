# Environment Setup Guide

This guide helps you configure the GitHub Repository Explorer for both Azure OpenAI and Ollama AI agents.

## Quick Setup

1. **Copy the environment template:**
   ```bash
   cp .env.template .env
   ```

2. **Configure your preferred AI agent** (see sections below)

3. **Start the application:**
   ```bash
   make build_run_clean
   ```

## Azure OpenAI Setup (Cloud-based AI)

### Prerequisites
- Azure subscription
- Azure OpenAI resource with GPT-4o deployment

### Configuration
Add these variables to your `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
REACT_AGENT_TYPE=azure
```

### Getting Azure OpenAI Credentials

1. Go to [Azure Portal](https://portal.azure.com)
2. Create or navigate to your Azure OpenAI resource
3. Deploy a GPT-4o model
4. Go to "Keys and Endpoint" section
5. Copy the API key and endpoint

### Pros/Cons
✅ **Pros:** Best performance, fast responses, excellent reasoning  
❌ **Cons:** Requires internet, usage costs, needs Azure account

## Ollama Setup (Local AI)

### Prerequisites
- Local machine with sufficient RAM (8GB+ recommended)
- Ollama installed and running

### Installation Steps

1. **Install Ollama:**
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Or download from https://ollama.com/download
   ```

2. **Pull a model (or use existing):**
   ```bash
   # Best for code analysis (recommended for this app):
   ollama pull qwen2.5-coder:7b-instruct
   
   # If you already have models, check with:
   ollama list
   
   # Other great options for code:
   ollama pull qwen2.5-coder:32b-instruct  # Higher quality, needs more RAM
   ollama pull codellama:7b                # Alternative code model
   ```

3. **Start Ollama service:**
   ```bash
   ollama serve
   ```

### Configuration
Add these variables to your `.env` file:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5-coder:7b-instruct
REACT_AGENT_TYPE=ollama
```

### Docker Users
If running in Docker, use this configuration:

```bash
# For Docker containers
OLLAMA_HOST=http://host.docker.internal:11434
```

### Model Recommendations

| Model | Size | RAM Required | Best For |
|-------|------|-------------|----------|
| `qwen2.5-coder:7b-instruct` | 4.7GB | 8GB+ | **Recommended** - Excellent for code |
| `qwen2.5-coder:32b-instruct-q8_0` | 34GB | 40GB+ | Best code analysis quality |
| `llama3.1:70b-instruct-q4_K_M` | 42GB | 48GB+ | Best reasoning (if you have RAM) |
| `llama3.1:8b-instruct-q4_K_M` | 4.9GB | 8GB+ | Good general purpose |
| `qwq:latest` | 19GB | 24GB+ | Strong reasoning model |
| `llama3.2:3b-instruct-q8_0` | 3.4GB | 6GB+ | Fastest option |

### Pros/Cons
✅ **Pros:** Local/private, no costs, offline capable  
❌ **Cons:** Slower, requires local resources, setup complexity

## Troubleshooting

### Azure OpenAI Issues

**Problem:** "Azure OpenAI agent not functional"  
**Solution:** 
- Check your API key and endpoint are correct
- Verify the model deployment name matches `AZURE_OPENAI_MODEL_NAME`
- Ensure your Azure subscription has quota

**Problem:** "Missing environment variables"  
**Solution:** 
- Ensure all `AZURE_OPENAI_*` variables are set in your `.env` file
- Restart the Docker containers after changing environment variables

### Ollama Issues

**Problem:** "Failed to connect to Ollama"  
**Solution:**
1. Verify Ollama is running: `ollama list`
2. Check the service: `ollama serve`
3. Test connection: `curl http://localhost:11434/api/tags`
4. For Docker: ensure `host.docker.internal` is accessible

**Problem:** "Model not found"  
**Solution:**
- Pull the model: `ollama pull llama3.1:8b`
- Check available models: `ollama list`
- Update `OLLAMA_MODEL_NAME` to match an available model

**Problem:** "Permission denied" or "Connection refused"  
**Solution:**
- Ensure Ollama service is running as your user
- Check firewall settings (port 11434)
- For Docker: verify extra_hosts configuration

### General Issues

**Problem:** Application falls back to traditional mode  
**Solution:**
- Check the logs for specific error messages
- Verify environment variables are set correctly
- Try switching agents in the UI Settings panel

**Problem:** "No ReAct agent available"  
**Solution:**
- Ensure at least one agent (Azure or Ollama) is properly configured
- Check the Agent Status section in the sidebar
- Restart the application after configuration changes

## Agent Comparison

| Feature | Azure OpenAI | Ollama |
|---------|-------------|---------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Privacy** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Ease** | ⭐⭐⭐⭐ | ⭐⭐ |
| **Offline Use** | ❌ | ✅ |
| **Response Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Environment Template

Create a `.env` file with this template:

```bash
# =============================================================================
# Choose your agent: "azure" or "ollama"
# =============================================================================
REACT_AGENT_TYPE=azure

# =============================================================================
# Azure OpenAI Configuration (for cloud-based AI)
# =============================================================================
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# =============================================================================
# Ollama Configuration (for local AI)
# =============================================================================
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_NAME=qwen2.5-coder:7b-instruct

# =============================================================================
# Repository Configuration
# =============================================================================
GITHUB_REPOS_ROOT=/home/snow/Documents/Projects/github-repositories/bkocis
MCP_SERVER_URL=http://localhost:8081/mcp
```

## Testing Your Setup

1. **Start the application:**
   ```bash
   make build_run_clean
   ```

2. **Check the sidebar** for agent status indicators

3. **Test with a simple query:** "List all README files"

4. **Verify the correct agent is being used** in the execution metrics

Need help? Check the application logs with `make logs` or open an issue in the repository.