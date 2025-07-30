# Ollama Models Guide for Your GitHub Repository Explorer

Based on your available models, here are the best options for analyzing GitHub repositories:

## 🏆 Recommended Models (Best to Good)

### 1. **qwen2.5-coder:7b-instruct** ⭐⭐⭐⭐⭐
```bash
# Set in your .env file:
OLLAMA_MODEL_NAME=qwen2.5-coder:7b-instruct
```
- **Size:** 4.7 GB
- **Why it's best:** Latest code-specialized model, excellent for repository analysis
- **Perfect for:** Understanding code structure, documentation analysis, technology identification

### 2. **qwen2.5-coder:32b-instruct-q8_0** ⭐⭐⭐⭐⭐ (if you have RAM)
```bash
# Set in your .env file:
OLLAMA_MODEL_NAME=qwen2.5-coder:32b-instruct-q8_0
```
- **Size:** 34 GB (requires 40GB+ RAM)
- **Why it's great:** Highest quality code analysis
- **Perfect for:** Complex repository analysis, detailed code understanding

### 3. **llama3.1:70b-instruct-q4_K_M** ⭐⭐⭐⭐⭐ (for reasoning)
```bash
# Set in your .env file:
OLLAMA_MODEL_NAME=llama3.1:70b-instruct-q4_K_M
```
- **Size:** 42 GB (requires 48GB+ RAM)
- **Why it's great:** Best reasoning capabilities
- **Perfect for:** Complex questions about repository relationships and patterns

### 4. **qwq:latest** ⭐⭐⭐⭐ (reasoning specialist)
```bash
# Set in your .env file:
OLLAMA_MODEL_NAME=qwq:latest
```
- **Size:** 19 GB (requires 24GB+ RAM)
- **Why it's good:** Strong reasoning model
- **Perfect for:** Understanding project purposes and connections

### 5. **llama3.1:8b-instruct-q4_K_M** ⭐⭐⭐ (balanced option)
```bash
# Set in your .env file:
OLLAMA_MODEL_NAME=llama3.1:8b-instruct-q4_K_M
```
- **Size:** 4.9 GB
- **Why it's decent:** Good general-purpose model
- **Perfect for:** Basic repository exploration with moderate resource usage

## 🚀 Quick Setup Commands

### Option 1: Use the best model (recommended)
```bash
# Create/edit your .env file
echo "OLLAMA_MODEL_NAME=qwen2.5-coder:7b-instruct" >> .env
echo "REACT_AGENT_TYPE=ollama" >> .env

# Restart the application
make build_run_clean
```

### Option 2: Use the highest quality (if you have 40GB+ RAM)
```bash
# Create/edit your .env file
echo "OLLAMA_MODEL_NAME=qwen2.5-coder:32b-instruct-q8_0" >> .env
echo "REACT_AGENT_TYPE=ollama" >> .env

# Restart the application
make build_run_clean
```

### Option 3: Use the fastest option
```bash
# Create/edit your .env file
echo "OLLAMA_MODEL_NAME=llama3.2:3b-instruct-q8_0" >> .env
echo "REACT_AGENT_TYPE=ollama" >> .env

# Restart the application
make build_run_clean
```

## 🧪 Testing Different Models

You can easily switch between models in the Streamlit UI:

1. **Go to the sidebar** → "🎯 Model Selection"
2. **Select "🏠 Ollama (Local LLM)"**
3. **Expand "🔧 Ollama Configuration"**
4. **Change the model name** to any of your available models
5. **Click "🔄 Update Ollama Config"**

## 📊 Performance Comparison

| Model | Speed | Code Quality | Reasoning | RAM Usage |
|-------|-------|-------------|-----------|-----------|
| qwen2.5-coder:7b-instruct | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low |
| qwen2.5-coder:32b-instruct | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Very High |
| llama3.1:70b-instruct | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Very High |
| qwq:latest | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High |
| llama3.1:8b-instruct | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Low |
| llama3.2:3b-instruct | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Very Low |

## 🎯 Model Selection Guide

**Choose based on your system:**

- **8-16 GB RAM:** Use `qwen2.5-coder:7b-instruct` or `llama3.2:3b-instruct-q8_0`
- **24-32 GB RAM:** Use `qwq:latest` for reasoning tasks
- **40+ GB RAM:** Use `qwen2.5-coder:32b-instruct-q8_0` for best code analysis
- **48+ GB RAM:** Use `llama3.1:70b-instruct-q4_K_M` for best overall quality

**Choose based on your priority:**

- **Best for code analysis:** `qwen2.5-coder:7b-instruct` or `qwen2.5-coder:32b-instruct-q8_0`
- **Best for reasoning:** `qwq:latest` or `llama3.1:70b-instruct-q4_K_M`
- **Fastest responses:** `llama3.2:3b-instruct-q8_0`
- **Best balance:** `qwen2.5-coder:7b-instruct`

## 🔧 Troubleshooting

**Model not working?**
```bash
# Check if the model is available
ollama list | grep "your-model-name"

# Test the model directly
ollama run qwen2.5-coder:7b-instruct "Hello, can you analyze code?"

# Check Ollama service
ollama ps
```

**Out of memory errors?**
- Switch to a smaller model like `llama3.2:3b-instruct-q8_0`
- Close other applications to free up RAM
- Use quantized versions (models with `q4_K_M` or `q8_0` in the name)

Your current setup is already configured to use **qwen2.5-coder:7b-instruct** which is the best choice for this application! 🎉