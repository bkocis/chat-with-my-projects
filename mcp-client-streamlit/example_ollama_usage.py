#!/usr/bin/env python3
"""
Example script demonstrating the usage of both Azure OpenAI and Ollama ReAct agents.
"""

import os
from mcp_client import MCPClient, EnhancedQueryProcessor

def main():
    print("=== MCP Client ReAct Agent Demo ===\n")
    
    # Initialize MCP client
    print("Initializing MCP client...")
    client = MCPClient()
    
    if not client.initialize():
        print("Failed to initialize MCP client. Please ensure the server is running.")
        return
    
    # Test query
    test_query = "Find the best credit cards for travel with no annual fee"
    
    print(f"Test Query: {test_query}\n")
    
    # Test Azure OpenAI agent (if available)
    print("=" * 50)
    print("Testing Azure OpenAI ReAct Agent")
    print("=" * 50)
    
    azure_processor = EnhancedQueryProcessor(client, use_react=True, agent_type="azure")
    azure_info = azure_processor.get_agent_info()
    
    print(f"Agent Type: {azure_info['type']}")
    print(f"Available: {azure_info['available']}")
    print(f"Model: {azure_info.get('model', 'N/A')}")
    
    if azure_info['available']:
        print("\nProcessing query with Azure agent...")
        try:
            result, trace = azure_processor.process_query_with_trace(test_query)
            print(f"Result: {result[:200]}...")
            print(f"Execution steps: {len(trace) if trace else 0}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Azure OpenAI not available (check environment variables)")
    
    print("\n" + "=" * 50)
    print("Testing Ollama ReAct Agent")
    print("=" * 50)
    
    # Test Ollama agent
    ollama_processor = EnhancedQueryProcessor(client, use_react=True, agent_type="ollama")
    ollama_info = ollama_processor.get_agent_info()
    
    print(f"Agent Type: {ollama_info['type']}")
    print(f"Available: {ollama_info['available']}")
    print(f"Model: {ollama_info.get('model', 'N/A')}")
    print(f"Host: {ollama_info.get('host', 'N/A')}")
    
    if ollama_info['available']:
        print("\nProcessing query with Ollama agent...")
        try:
            result, trace = ollama_processor.process_query_with_trace(test_query)
            print(f"Result: {result[:200]}...")
            print(f"Execution steps: {len(trace) if trace else 0}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Ollama not available (install ollama package and ensure Ollama server is running)")
    
    print("\n" + "=" * 50)
    print("Testing Agent Switching")
    print("=" * 50)
    
    # Demonstrate agent switching
    processor = EnhancedQueryProcessor(client, use_react=True, agent_type="azure")
    print(f"Started with: {processor.get_agent_info()['type']}")
    
    if processor.switch_agent_type("ollama"):
        print(f"Switched to: {processor.get_agent_info()['type']}")
    else:
        print("Failed to switch to Ollama")
    
    if processor.switch_agent_type("azure"):
        print(f"Switched back to: {processor.get_agent_info()['type']}")
    
    print("\n" + "=" * 50)
    print("Testing Fallback to Traditional Processing")
    print("=" * 50)
    
    # Test traditional processing (no ReAct)
    traditional_processor = EnhancedQueryProcessor(client, use_react=False)
    print("Processing query with traditional logic...")
    result = traditional_processor.process_query(test_query)
    print(f"Result: {result[:200]}...")

def setup_environment():
    """Example of setting up environment variables"""
    print("Example environment setup:")
    print("export REACT_AGENT_TYPE=ollama")
    print("export OLLAMA_MODEL_NAME=llama3.1:8b")
    print("export OLLAMA_HOST=http://localhost:11434")
    print()
    print("For Azure OpenAI:")
    print("export AZURE_OPENAI_API_KEY=your_key")
    print("export AZURE_OPENAI_ENDPOINT=your_endpoint")
    print("export AZURE_OPENAI_MODEL_NAME=gpt-4o-0513-eu")
    print()

if __name__ == "__main__":
    # Check if help is requested
    if len(os.sys.argv) > 1 and os.sys.argv[1] in ["-h", "--help"]:
        setup_environment()
    else:
        main() 