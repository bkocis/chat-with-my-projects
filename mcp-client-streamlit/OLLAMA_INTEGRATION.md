# Ollama ReAct Agent Integration

This document explains how to use the new Ollama-based ReAct agent alongside the existing Azure OpenAI implementation.

## Prerequisites

1. **Install Ollama**: Download and install Ollama from [https://ollama.ai](https://ollama.ai)
2. **Pull a model**: Run `ollama pull llama3.1:8b` (or your preferred model)
3. **Install Python dependencies**: `pip install ollama>=0.1.0`

## Environment Variables

Configure these environment variables to use Ollama:

```bash
# Choose the agent type (azure or ollama)
export REACT_AGENT_TYPE=ollama

# Ollama-specific settings
export OLLAMA_MODEL_NAME=llama3.1:8b  # default model
export OLLAMA_HOST=http://localhost:11434  # default Ollama server
```

## Usage Examples

### Basic Usage

```python
from mcp_client import MCPClient, EnhancedQueryProcessor

# Initialize MCP client
client = MCPClient()

# Use Ollama agent
processor = EnhancedQueryProcessor(client, use_react=True, agent_type="ollama")

# Process queries
result = processor.process_query("Find the best travel credit cards under €50 annual fee")
```

### Switching Between Agents

```python
# Start with Azure (default)
processor = EnhancedQueryProcessor(client, use_react=True, agent_type="azure")

# Switch to Ollama at runtime
processor.switch_agent_type("ollama")

# Get agent information
info = processor.get_agent_info()
print(f"Current agent: {info['type']}, Available: {info['available']}")
```

### With Execution Trace

```python
# Get both result and execution trace
result, trace = processor.process_query_with_trace("Compare Chase Sapphire cards")

# Inspect the reasoning steps
for step in trace:
    print(f"Iteration {step['iteration']}: {step['reasoning'][:100]}...")
```

## Recommended Models

For best results with credit card queries, we recommend:

- **llama3.1:8b** - Good balance of speed and quality
- **llama3.1:13b** - Better reasoning, slower
- **mistral:7b** - Faster, good for simple queries
- **gemma2:9b** - Alternative option

## Troubleshooting

### Model Not Found
```bash
# Pull the model first
ollama pull llama3.1:8b

# Or use a different available model
export OLLAMA_MODEL_NAME=mistral:7b
```

### Connection Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve
```

### Performance Tips

1. **Use smaller models** for faster responses
2. **Adjust temperature** in the agent for more/less creative responses
3. **Increase max_iterations** for complex queries that need multiple tool calls

## Backwards Compatibility

The integration is fully backwards compatible:

- Existing code using `EnhancedQueryProcessor` will continue to work with Azure OpenAI
- Set `REACT_AGENT_TYPE=azure` or omit the variable to use Azure
- The system gracefully falls back to traditional processing if agents fail

## Benefits of Ollama Integration

1. **Privacy**: All processing happens locally
2. **Cost**: No API costs for LLM calls
3. **Speed**: No network latency for model calls
4. **Offline**: Works without internet connection
5. **Control**: Full control over model selection and parameters 