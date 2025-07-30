#!/usr/bin/env python3
"""
MCP HTTP Client for GitHub Repositories Server

This client connects to the GitHub repositories MCP server and provides
an easy interface for calling tools and reading resources.
Enhanced with Azure OpenAI GPT-4o and Ollama for ReAct (Reasoning and Acting) capabilities.
"""

import json
import requests
import uuid
import os
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI

# Try to import ollama, make it optional
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class MCPClient:
    """HTTP-based MCP client for the GitHub repositories server"""
    
    def __init__(self, server_url: str = None):
        if server_url is None:
            # Use environment variable or default to localhost for development
            server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
        self.server_url = server_url
        self.session_id = None
        self.session = requests.Session()
        
    def initialize(self) -> bool:
        """Initialize the MCP session"""
        try:
            response = self.session.post(
                self.server_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "streamlit-github-repos-client",
                            "version": "1.0.0"
                        }
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                self.session_id = response.headers.get("mcp-session-id")
                if self.session_id:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Failed to initialize MCP session: {e}")
            return False
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call a tool on the MCP server"""
        if not self.session_id:
            if not self.initialize():
                return "Error: Could not establish MCP session"
        
        try:
            response = self.session.post(
                self.server_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "id": uuid.uuid4().hex[:8],
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "mcp-session-id": self.session_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "content" in result["result"]:
                    return result["result"]["content"][0]["text"]
                elif "error" in result:
                    return f"Error: {result['error']['message']}"
            
            return f"Error: HTTP {response.status_code}"
            
        except Exception as e:
            return f"Error calling tool: {e}"
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        if not self.session_id:
            if not self.initialize():
                return []
        
        try:
            response = self.session.post(
                self.server_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": uuid.uuid4().hex[:8],
                    "params": {}
                },
                headers={
                    "Content-Type": "application/json",
                    "mcp-session-id": self.session_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "tools" in result["result"]:
                    return result["result"]["tools"]
            
            return []
            
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []
    
    def get_debug_status(self) -> str:
        """Get debug status from the server (legacy method)"""
        return self.call_tool("debug_data_status", {})


class QueryProcessor:
    """Process natural language queries and determine appropriate MCP tool calls for GitHub repositories"""
    
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
    
    def process_query(self, query: str) -> str:
        """Process a natural language query and return appropriate response"""
        query_lower = query.lower()
        
        # Debug/status queries
        if any(word in query_lower for word in ["status", "debug", "data", "loaded", "available"]):
            return self.client.call_tool("get_technology_statistics", {})
        
        # Language-specific searches
        if any(word in query_lower for word in ["python", "javascript", "java", "typescript", "go", "rust", "c++", "c#"]):
            # Extract language name
            languages = ["python", "javascript", "java", "typescript", "go", "rust", "c++", "c#", "php", "ruby", "swift", "kotlin"]
            for lang in languages:
                if lang in query_lower:
                    return self.client.call_tool("search_repositories_by_language", {"language": lang})
        
        # Framework-specific searches
        if any(word in query_lower for word in ["react", "django", "flask", "express", "spring", "laravel", "rails"]):
            frameworks = ["react", "django", "flask", "express", "spring", "laravel", "rails", "vue", "angular", "tensorflow", "pytorch"]
            for framework in frameworks:
                if framework in query_lower:
                    return self.client.call_tool("search_repositories_by_framework", {"framework": framework})
        
        # README content searches
        if any(word in query_lower for word in ["search", "find", "contains", "about"]):
            # Extract search term (simple approach)
            words = query.split()
            if len(words) > 1:
                search_term = " ".join(words[1:])  # Skip the first word (search, find, etc.)
                return self.client.call_tool("search_readme_content", {"search_term": search_term, "max_results": 10})
        
        # Repository details
        if any(word in query_lower for word in ["details", "info", "information"]) and "repository" in query_lower:
            # Try to extract repository name
            words = query.split()
            repo_words = []
            for word in words:
                if word.lower() not in ["details", "info", "information", "repository", "repo", "about", "get", "show"]:
                    repo_words.append(word)
            if repo_words:
                repo_name = "-".join(repo_words)
                return self.client.call_tool("get_repository_details", {"repository_name": repo_name})
        
        # List all README files
        if any(word in query_lower for word in ["list", "show", "all"]) and "readme" in query_lower:
            if "summary" in query_lower:
                return self.client.call_tool("get_all_readmes_summary", {})
            else:
                return self.client.call_tool("list_readme_files", {})
        
        # Technology statistics
        if any(word in query_lower for word in ["statistics", "stats", "overview", "technologies", "tech"]):
            return self.client.call_tool("get_technology_statistics", {})
        
        # Largest repositories
        if any(word in query_lower for word in ["largest", "biggest", "size"]):
            return self.client.call_tool("get_largest_repositories", {"limit": 10})
        
        # Default: treat as a README content search
        return self.client.call_tool("search_readme_content", {"search_term": query, "max_results": 10})



class ReActAgent:
    """
    Reasoning and Acting agent using Azure OpenAI GPT-4o model.
    Implements the ReAct pattern for intelligent GitHub repository exploration.
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.azure_client = None
        self.conversation_context = []  # Store execution trace
        self._initialize_azure_client()
        
        # Available tools description for the model
        self.tools_description = {
            "list_readme_files": "List all README files in all repositories (filesystem scan)",
            "get_readme_content": "Get the content of a specific README file (path)",
            "get_all_readmes_content": "Get the content of all README files as a list of {path, content} objects",
            "get_all_readmes_summary": "Get a summary of all README files as a list of {path, summary} objects",
            "search_repositories_by_language": "Find repositories that use a specific programming language (language)",
            "search_repositories_by_framework": "Find repositories that use a specific framework or technology (framework)",
            "search_readme_content": "Search across all README files for specific content (search_term, max_results)",
            "get_technology_statistics": "Get comprehensive statistics about technology usage across all repositories",
            "get_repository_recommendations": "Get recommendations for repositories similar to a given repository (repository_name, limit)",
            "get_largest_repositories": "Get the largest repositories by file size with their technology details (limit)",
            "get_repository_details": "Get comprehensive details about a specific repository (repository_name)"
        }
    
    def _initialize_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            # Check for required environment variables
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            if not api_key or not endpoint:
                print("Azure OpenAI credentials missing. Required: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
                self.azure_client = None
                return
            
            self.azure_client = AzureOpenAI(
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=endpoint
            )
            self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-0513-eu")
            print(f"Azure OpenAI client initialized successfully with model: {self.model_name}")
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI client: {e}")
            self.azure_client = None
    
    def reason_and_act(self, user_query: str, max_iterations: int = 3) -> str:
        """
        Main ReAct loop: Reason about the query, act with tools, and provide final answer.
        Returns the final answer and stores execution trace in self.conversation_context.
        """
        if not self.azure_client:
            return "Error: Azure OpenAI client not available. Please check your environment variables."
        
        # Reset conversation context for new query
        self.conversation_context = []
        final_result = ""
        
        for iteration in range(max_iterations):
            try:
                # Reasoning phase
                reasoning_prompt = self._build_reasoning_prompt(user_query, self.conversation_context, iteration)
                
                reasoning_response = self.azure_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    max_completion_tokens=4000
                )
                
                reasoning_text = reasoning_response.choices[0].message.content
                
                # Parse the reasoning to extract actions
                actions = self._parse_actions_from_reasoning(reasoning_text)
                
                if not actions:
                    # No more actions needed, provide final answer
                    final_result = self._generate_final_answer(user_query, self.conversation_context, reasoning_text)
                    break
                
                # Acting phase - execute the planned actions
                action_results = []
                for action in actions:
                    result = self._execute_action(action)
                    action_results.append({
                        "action": action,
                        "result": result
                    })
                
                # Add to conversation context
                self.conversation_context.append({
                    "iteration": iteration + 1,
                    "reasoning": reasoning_text,
                    "actions": actions,
                    "results": action_results
                })
                
                # Check if we have enough information for a final answer
                if self._has_sufficient_information(action_results):
                    final_result = self._generate_final_answer(user_query, self.conversation_context)
                    break
                    
            except Exception as e:
                error_msg = f"Error in ReAct iteration {iteration + 1}: {e}"
                # Add error to conversation context
                self.conversation_context.append({
                    "iteration": iteration + 1,
                    "reasoning": f"Error occurred: {e}",
                    "actions": [],
                    "results": []
                })
                return error_msg
        
        return final_result or "Unable to process query within maximum iterations."
    
    def get_execution_trace(self) -> List[Dict]:
        """Get the execution trace from the last query"""
        return self.conversation_context.copy()
    
    def _build_reasoning_prompt(self, user_query: str, context: List[Dict], iteration: int) -> str:
        """Build the reasoning prompt for the o1 model"""
        
        context_text = ""
        if context:
            context_text = "\n\n## Previous Actions and Results:\n"
            for ctx in context:
                context_text += f"\n### Iteration {ctx['iteration']}:\n"
                context_text += f"**Reasoning:** {ctx['reasoning']}\n"
                for action_result in ctx['results']:
                    context_text += f"**Action:** {action_result['action']}\n"
                    context_text += f"**Result:** {action_result['result'][:500]}...\n\n"
        
        prompt = f"""
You are an intelligent GitHub repository assistant that helps users explore and analyze GitHub repositories. 
You have access to the following tools:

{json.dumps(self.tools_description, indent=2)}

## User Query:
{user_query}

{context_text}

## Your Task:
Based on the user query{'and previous context' if context else ''}, reason about what actions you need to take.

If this is iteration {iteration + 1} and you need to gather more information, specify the tool calls you want to make in this format:
ACTION: tool_name
ARGS: {{"arg1": "value1", "arg2": "value2"}}

If you have sufficient information to answer the user's query, write:
FINAL_ANSWER: [your complete response to the user]

Think step by step about:
1. What information the user is asking for about GitHub repositories
2. What tools you need to use to get that information
3. How to combine the results to provide a helpful answer

Be specific and actionable in your reasoning.
"""
        
        return prompt
    
    def _parse_actions_from_reasoning(self, reasoning_text: str) -> List[Dict[str, Any]]:
        """Parse action commands from the reasoning text"""
        actions = []
        lines = reasoning_text.split('\n')
        
        current_action = None
        for line in lines:
            line = line.strip()
            if line.startswith('ACTION:'):
                if current_action:
                    actions.append(current_action)
                current_action = {"tool": line[7:].strip(), "args": {}}
            elif line.startswith('ARGS:') and current_action:
                try:
                    args_text = line[5:].strip()
                    current_action["args"] = json.loads(args_text)
                except json.JSONDecodeError:
                    # Try to parse as simple key=value pairs
                    current_action["args"] = {}
            elif line.startswith('FINAL_ANSWER:'):
                # No more actions needed
                break
        
        if current_action:
            actions.append(current_action)
        
        return actions
    
    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute a single action using the MCP client"""
        tool_name = action.get("tool", "")
        args = action.get("args", {})
        
        try:
            if tool_name == "list_readme_files":
                return self.mcp_client.call_tool("list_readme_files", {})
            elif tool_name == "get_readme_content":
                return self.mcp_client.call_tool("get_readme_content", {
                    "path": args.get("path", "")
                })
            elif tool_name == "get_all_readmes_content":
                return self.mcp_client.call_tool("get_all_readmes_content", {})
            elif tool_name == "get_all_readmes_summary":
                return self.mcp_client.call_tool("get_all_readmes_summary", {})
            elif tool_name == "search_repositories_by_language":
                return self.mcp_client.call_tool("search_repositories_by_language", {
                    "language": args.get("language", "")
                })
            elif tool_name == "search_repositories_by_framework":
                return self.mcp_client.call_tool("search_repositories_by_framework", {
                    "framework": args.get("framework", "")
                })
            elif tool_name == "search_readme_content":
                return self.mcp_client.call_tool("search_readme_content", {
                    "search_term": args.get("search_term", ""),
                    "max_results": args.get("max_results", 20)
                })
            elif tool_name == "get_technology_statistics":
                return self.mcp_client.call_tool("get_technology_statistics", {})
            elif tool_name == "get_repository_recommendations":
                return self.mcp_client.call_tool("get_repository_recommendations", {
                    "repository_name": args.get("repository_name", ""),
                    "limit": args.get("limit", 5)
                })
            elif tool_name == "get_largest_repositories":
                return self.mcp_client.call_tool("get_largest_repositories", {
                    "limit": args.get("limit", 10)
                })
            elif tool_name == "get_repository_details":
                return self.mcp_client.call_tool("get_repository_details", {
                    "repository_name": args.get("repository_name", "")
                })
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
    
    def _has_sufficient_information(self, action_results: List[Dict]) -> bool:
        """Check if we have enough information to provide a final answer"""
        # Simple heuristic: if we have at least one successful result, we can provide an answer
        for result in action_results:
            if not result["result"].startswith("Error"):
                return True
        return False
    
    def _generate_final_answer(self, user_query: str, context: List[Dict], reasoning_text: str = "") -> str:
        """Generate the final answer using the o1 model"""
        try:
            # Check if reasoning_text already contains a FINAL_ANSWER
            if "FINAL_ANSWER:" in reasoning_text:
                final_answer_start = reasoning_text.find("FINAL_ANSWER:") + 13
                return reasoning_text[final_answer_start:].strip()
            
            # Otherwise, generate a final answer based on context
            context_summary = ""
            for ctx in context:
                for action_result in ctx['results']:
                    context_summary += f"**{action_result['action']['tool']}**: {action_result['result'][:500]}\n\n"
            
            final_prompt = f"""
Based on the user query and the information gathered from the GitHub repositories, provide a comprehensive and helpful answer.

## User Query:
{user_query}

## Information Gathered:
{context_summary}

Provide a clear, well-formatted response that directly answers the user's question. Include specific details about repositories, their technologies, features, and characteristics where relevant.
"""
            
            response = self.azure_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": final_prompt}],
                max_completion_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating final answer: {e}"


class OllamaReActAgent:
    """
    Reasoning and Acting agent using Ollama local models.
    Implements the ReAct pattern for intelligent GitHub repository exploration with local LLMs.
    """
    
    def __init__(self, mcp_client: MCPClient, model_name: str = None, ollama_host: str = None):
        self.mcp_client = mcp_client
        self.model_name = model_name or os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_available = OLLAMA_AVAILABLE
        self.conversation_context = []  # Store execution trace
        
        # Available tools description for the model
        self.tools_description = {
            "list_readme_files": "List all README files in all repositories (filesystem scan)",
            "get_readme_content": "Get the content of a specific README file (path)",
            "get_all_readmes_content": "Get the content of all README files as a list of {path, content} objects",
            "get_all_readmes_summary": "Get a summary of all README files as a list of {path, summary} objects",
            "search_repositories_by_language": "Find repositories that use a specific programming language (language)",
            "search_repositories_by_framework": "Find repositories that use a specific framework or technology (framework)",
            "search_readme_content": "Search across all README files for specific content (search_term, max_results)",
            "get_technology_statistics": "Get comprehensive statistics about technology usage across all repositories",
            "get_repository_recommendations": "Get recommendations for repositories similar to a given repository (repository_name, limit)",
            "get_largest_repositories": "Get the largest repositories by file size with their technology details (limit)",
            "get_repository_details": "Get comprehensive details about a specific repository (repository_name)"
        }
        
        # Test Ollama connectivity
        self._test_ollama_connection()
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is available and responsive"""
        if not self.ollama_available:
            print("Warning: Ollama package not installed. Install with: pip install ollama")
            return False
        
        try:
            # Test connection by listing models
            models = ollama.list()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if self.model_name not in model_names:
                print(f"Warning: Model '{self.model_name}' not found in Ollama. Available models: {model_names}")
                # Try to use the first available model
                if model_names:
                    self.model_name = model_names[0]
                    print(f"Using model: {self.model_name}")
                else:
                    print("No models available in Ollama. Please pull a model first.")
                    return False
            
            return True
        except Exception as e:
            print(f"Failed to connect to Ollama: {e}")
            return False
    
    def reason_and_act(self, user_query: str, max_iterations: int = 3) -> str:
        """
        Main ReAct loop: Reason about the query, act with tools, and provide final answer.
        Returns the final answer and stores execution trace in self.conversation_context.
        """
        if not self.ollama_available:
            return "Error: Ollama not available. Please install ollama package and ensure Ollama server is running."
        
        # Reset conversation context for new query
        self.conversation_context = []
        final_result = ""
        
        for iteration in range(max_iterations):
            try:
                # Reasoning phase
                reasoning_prompt = self._build_reasoning_prompt(user_query, self.conversation_context, iteration)
                
                reasoning_response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    options={
                        "temperature": 0.1,  # Lower temperature for more consistent reasoning
                        "top_p": 0.9,
                        "num_predict": 1000  # Limit response length
                    }
                )
                
                reasoning_text = reasoning_response['message']['content']
                
                # Parse the reasoning to extract actions
                actions = self._parse_actions_from_reasoning(reasoning_text)
                
                if not actions:
                    # No more actions needed, provide final answer
                    final_result = self._generate_final_answer(user_query, self.conversation_context, reasoning_text)
                    break
                
                # Acting phase - execute the planned actions
                action_results = []
                for action in actions:
                    result = self._execute_action(action)
                    action_results.append({
                        "action": action,
                        "result": result
                    })
                
                # Add to conversation context
                self.conversation_context.append({
                    "iteration": iteration + 1,
                    "reasoning": reasoning_text,
                    "actions": actions,
                    "results": action_results
                })
                
                # Check if we have enough information for a final answer
                if self._has_sufficient_information(action_results):
                    final_result = self._generate_final_answer(user_query, self.conversation_context)
                    break
                    
            except Exception as e:
                error_msg = f"Error in ReAct iteration {iteration + 1}: {e}"
                # Add error to conversation context
                self.conversation_context.append({
                    "iteration": iteration + 1,
                    "reasoning": f"Error occurred: {e}",
                    "actions": [],
                    "results": []
                })
                return error_msg
        
        return final_result or "Unable to process query within maximum iterations."
    
    def get_execution_trace(self) -> List[Dict]:
        """Get the execution trace from the last query"""
        return self.conversation_context.copy()
    
    def _build_reasoning_prompt(self, user_query: str, context: List[Dict], iteration: int) -> str:
        """Build the reasoning prompt for the Ollama model"""
        
        context_text = ""
        if context:
            context_text = "\n\n## Previous Actions and Results:\n"
            for ctx in context:
                context_text += f"\n### Iteration {ctx['iteration']}:\n"
                context_text += f"**Reasoning:** {ctx['reasoning']}\n"
                for action_result in ctx['results']:
                    context_text += f"**Action:** {action_result['action']}\n"
                    context_text += f"**Result:** {action_result['result'][:500]}...\n\n"
        
        prompt = f"""You are an intelligent GitHub repository assistant that helps users explore and analyze GitHub repositories. 
You have access to the following tools:

{json.dumps(self.tools_description, indent=2)}

## User Query:
{user_query}

{context_text}

## Your Task:
Based on the user query{'and previous context' if context else ''}, reason about what actions you need to take.

If this is iteration {iteration + 1} and you need to gather more information, specify the tool calls you want to make in this format:
ACTION: tool_name
ARGS: {{"arg1": "value1", "arg2": "value2"}}

If you have sufficient information to answer the user's query, write:
FINAL_ANSWER: [your complete response to the user]

Think step by step about:
1. What information the user is asking for about GitHub repositories
2. What tools you need to use to get that information
3. How to combine the results to provide a helpful answer

Be specific and actionable in your reasoning. Only provide one ACTION per response unless multiple actions are clearly needed.
"""
        
        return prompt
    
    def _parse_actions_from_reasoning(self, reasoning_text: str) -> List[Dict[str, Any]]:
        """Parse action commands from the reasoning text"""
        actions = []
        lines = reasoning_text.split('\n')
        
        current_action = None
        for line in lines:
            line = line.strip()
            if line.startswith('ACTION:'):
                if current_action:
                    actions.append(current_action)
                current_action = {"tool": line[7:].strip(), "args": {}}
            elif line.startswith('ARGS:') and current_action:
                try:
                    args_text = line[5:].strip()
                    current_action["args"] = json.loads(args_text)
                except json.JSONDecodeError:
                    # Try to parse as simple key=value pairs
                    current_action["args"] = {}
            elif line.startswith('FINAL_ANSWER:'):
                # No more actions needed
                break
        
        if current_action:
            actions.append(current_action)
        
        return actions
    
    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute a single action using the MCP client"""
        tool_name = action.get("tool", "")
        args = action.get("args", {})
        
        try:
            if tool_name == "list_readme_files":
                return self.mcp_client.call_tool("list_readme_files", {})
            elif tool_name == "get_readme_content":
                return self.mcp_client.call_tool("get_readme_content", {
                    "path": args.get("path", "")
                })
            elif tool_name == "get_all_readmes_content":
                return self.mcp_client.call_tool("get_all_readmes_content", {})
            elif tool_name == "get_all_readmes_summary":
                return self.mcp_client.call_tool("get_all_readmes_summary", {})
            elif tool_name == "search_repositories_by_language":
                return self.mcp_client.call_tool("search_repositories_by_language", {
                    "language": args.get("language", "")
                })
            elif tool_name == "search_repositories_by_framework":
                return self.mcp_client.call_tool("search_repositories_by_framework", {
                    "framework": args.get("framework", "")
                })
            elif tool_name == "search_readme_content":
                return self.mcp_client.call_tool("search_readme_content", {
                    "search_term": args.get("search_term", ""),
                    "max_results": args.get("max_results", 20)
                })
            elif tool_name == "get_technology_statistics":
                return self.mcp_client.call_tool("get_technology_statistics", {})
            elif tool_name == "get_repository_recommendations":
                return self.mcp_client.call_tool("get_repository_recommendations", {
                    "repository_name": args.get("repository_name", ""),
                    "limit": args.get("limit", 5)
                })
            elif tool_name == "get_largest_repositories":
                return self.mcp_client.call_tool("get_largest_repositories", {
                    "limit": args.get("limit", 10)
                })
            elif tool_name == "get_repository_details":
                return self.mcp_client.call_tool("get_repository_details", {
                    "repository_name": args.get("repository_name", "")
                })
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
    
    def _has_sufficient_information(self, action_results: List[Dict]) -> bool:
        """Check if we have enough information to provide a final answer"""
        # Simple heuristic: if we have at least one successful result, we can provide an answer
        for result in action_results:
            if not result["result"].startswith("Error"):
                return True
        return False
    
    def _generate_final_answer(self, user_query: str, context: List[Dict], reasoning_text: str = "") -> str:
        """Generate the final answer using the Ollama model"""
        try:
            # Check if reasoning_text already contains a FINAL_ANSWER
            if "FINAL_ANSWER:" in reasoning_text:
                final_answer_start = reasoning_text.find("FINAL_ANSWER:") + 13
                return reasoning_text[final_answer_start:].strip()
            
            # Otherwise, generate a final answer based on context
            context_summary = ""
            for ctx in context:
                for action_result in ctx['results']:
                    context_summary += f"**{action_result['action']['tool']}**: {action_result['result'][:500]}\n\n"
            
            final_prompt = f"""Based on the user query and the information gathered from the GitHub repositories, provide a comprehensive and helpful answer.

## User Query:
{user_query}

## Information Gathered:
{context_summary}

Provide a clear, well-formatted response that directly answers the user's question. Include specific details about repositories, their technologies, features, and characteristics where relevant.
"""
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": final_prompt}],
                options={
                    "temperature": 0.3,
                    "num_predict": 500
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error generating final answer: {e}"


class EnhancedQueryProcessor(QueryProcessor):
    """
    Enhanced query processor that uses ReAct agent for intelligent responses.
    Supports both Azure OpenAI and Ollama implementations.
    """
    
    def __init__(self, mcp_client: MCPClient, use_react: bool = True, agent_type: str = None):
        super().__init__(mcp_client)
        self.use_react = use_react
        
        # Determine agent type from parameter or environment variable
        self.agent_type = agent_type or os.getenv("REACT_AGENT_TYPE", "azure")  # default to azure for backwards compatibility
        
        # Initialize the appropriate agent
        self.react_agent = None
        self.last_execution_trace = None
        
        if use_react:
            self._initialize_react_agent()
    
    def _initialize_react_agent(self):
        """Initialize the appropriate ReAct agent based on agent_type"""
        try:
            if self.agent_type.lower() == "ollama":
                self.react_agent = OllamaReActAgent(self.client)
                if self.react_agent.ollama_available:
                    print(f"✅ Initialized Ollama ReAct agent with model: {self.react_agent.model_name}")
                else:
                    print("❌ Ollama ReAct agent failed to initialize - Ollama not available")
                    self.react_agent = None
            else:  # default to azure
                self.react_agent = ReActAgent(self.client)
                if self.react_agent.azure_client:
                    print(f"✅ Initialized Azure OpenAI ReAct agent with model: {self.react_agent.model_name}")
                else:
                    print("❌ Azure OpenAI ReAct agent failed to initialize - missing credentials")
                    self.react_agent = None
        except Exception as e:
            print(f"❌ Failed to initialize {self.agent_type} ReAct agent: {e}")
            self.react_agent = None
    
    def switch_agent_type(self, agent_type: str):
        """Switch between Azure and Ollama agents at runtime"""
        if agent_type.lower() in ["azure", "ollama"]:
            self.agent_type = agent_type.lower()
            if self.use_react:
                self._initialize_react_agent()
            return True
        return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent"""
        if not self.react_agent:
            return {"type": "none", "available": False}
        
        if isinstance(self.react_agent, OllamaReActAgent):
            return {
                "type": "ollama",
                "available": self.react_agent.ollama_available,
                "model": self.react_agent.model_name,
                "host": self.react_agent.ollama_host
            }
        else:
            return {
                "type": "azure",
                "available": self.react_agent.azure_client is not None,
                "model": getattr(self.react_agent, 'model_name', 'unknown')
            }
    
    def process_query(self, query: str, use_react: bool = None) -> str:
        """Process a query using either ReAct agent or traditional logic"""
        use_react = use_react if use_react is not None else self.use_react
        
        # Reset execution trace
        self.last_execution_trace = None
        
        if use_react:
            # Check if we have a properly initialized agent
            if not self.react_agent:
                print(f"⚠️  No ReAct agent available ({self.agent_type}), falling back to traditional processing")
                return super().process_query(query)
            
            # Check if the agent is properly functional
            agent_available = False
            if isinstance(self.react_agent, OllamaReActAgent):
                agent_available = self.react_agent.ollama_available
                if not agent_available:
                    print("⚠️  Ollama agent not functional, falling back to traditional processing")
            elif isinstance(self.react_agent, ReActAgent):
                agent_available = self.react_agent.azure_client is not None
                if not agent_available:
                    print("⚠️  Azure OpenAI agent not functional, falling back to traditional processing")
            
            if agent_available:
                # Use ReAct agent
                print(f"🤖 Using {self.agent_type.title()} ReAct agent for query processing")
                result = self.react_agent.reason_and_act(query)
                self.last_execution_trace = self.react_agent.get_execution_trace()
                return result
            else:
                return super().process_query(query)
        else:
            # Use traditional processing
            print("📝 Using traditional query processing")
            return super().process_query(query)
    
    def process_query_with_trace(self, query: str, use_react: bool = None) -> Tuple[str, Optional[List[Dict]]]:
        """Process a query and return both result and execution trace"""
        result = self.process_query(query, use_react)
        return result, self.last_execution_trace
    
    def get_last_execution_trace(self) -> Optional[List[Dict]]:
        """Get the execution trace from the last query"""
        return self.last_execution_trace 