#!/usr/bin/env python3
"""
MCP HTTP Client for GitHub Repositories Server

This client connects to the GitHub repositories MCP server and provides
an easy interface for calling tools and reading resources.
Enhanced with Azure OpenAI for ReAct (Reasoning and Acting) capabilities.
"""

import json
import requests
import uuid
import os
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI


class MCPClient:
    """HTTP-based MCP client for the GitHub repositories server"""
    
    def __init__(self, server_url: str = None):
        if server_url is None:
            # Use environment variable or default to localhost for development
            server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8081/mcp")
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
    
    def list_readme_files(self) -> str:
        """List all README files in all repositories"""
        return self.call_tool("list_readme_files", {})
    
    def get_readme_content(self, path: str) -> str:
        """Get the content of a specific README file"""
        return self.call_tool("get_readme_content", {"path": path})
    
    def get_all_readmes_content(self) -> str:
        """Get the content of all README files as a list of {path, content} objects"""
        return self.call_tool("get_all_readmes_content", {})
    
    def get_all_readmes_summary(self) -> str:
        """Get a summary (first 10 lines or 1000 chars) of all README files"""
        return self.call_tool("get_all_readmes_summary", {})


class QueryProcessor:
    """Process natural language queries and determine appropriate MCP tool calls"""
    
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
    
    def process_query(self, query: str) -> str:
        """Process a natural language query and return appropriate response"""
        query_lower = query.lower()
        
        # List README files queries
        if any(word in query_lower for word in ["list", "show", "find"]) and "readme" in query_lower and "files" in query_lower:
            return self.client.list_readme_files()
        
        # Get all README content
        if any(word in query_lower for word in ["all", "full"]) and "content" in query_lower and "readme" in query_lower:
            return self.client.get_all_readmes_content()
        
        # Get README summaries
        if "summary" in query_lower or "summaries" in query_lower:
            return self.client.get_all_readmes_summary()
        
        # Get specific README content
        if "readme" in query_lower and any(word in query_lower for word in ["content", "show", "get"]) and "specific" in query_lower:
            # This would need a path parameter - for now return instruction
            return "Please use the 'Get Specific README' tool in Advanced Options to specify a file path."
        
        # General repository exploration queries
        if any(word in query_lower for word in ["repositories", "repos", "projects", "documentation"]):
            # Default to showing summaries for general exploration
            return self.client.get_all_readmes_summary()
        
        # Technology/language specific queries
        if any(tech in query_lower for tech in ["python", "javascript", "java", "c++", "golang", "rust", "machine learning", "ai", "data"]):
            # Get all content to search through
            return self.client.get_all_readmes_content()
        
        # Default: get README summaries for any other query
        return self.client.get_all_readmes_summary()
    



class ReActAgent:
    """
    Reasoning and Acting agent using Azure OpenAI.
    Implements the ReAct pattern for intelligent tool usage.
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.azure_client = None
        self.conversation_context = []  # Store execution trace
        self._initialize_azure_client()
        
        # Available tools description for the AI model
        self.tools_description = {
            "list_readme_files": "List all README files in all GitHub repositories",
            "get_readme_content": "Get the content of a specific README file (path)",
            "get_all_readmes_content": "Get the content of all README files as a list of {path, content} objects",
            "get_all_readmes_summary": "Get a summary (first 10 lines or 1000 chars) of all README files"
        }
    
    def _initialize_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            self.azure_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o")
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
You are an intelligent GitHub repository assistant that helps users explore and understand GitHub repositories. 
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
1. What information the user is asking for about the repositories
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
                return self.mcp_client.list_readme_files()
            elif tool_name == "get_readme_content":
                return self.mcp_client.get_readme_content(args.get("path", ""))
            elif tool_name == "get_all_readmes_content":
                return self.mcp_client.get_all_readmes_content()
            elif tool_name == "get_all_readmes_summary":
                return self.mcp_client.get_all_readmes_summary()
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

Provide a clear, well-formatted response that directly answers the user's question. Include specific details about repositories, their purposes, technologies used, and any other relevant information.
"""
            
            response = self.azure_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": final_prompt}],
                max_completion_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating final answer: {e}"


class EnhancedQueryProcessor(QueryProcessor):
    """
    Enhanced query processor that uses ReAct agent for intelligent responses
    """
    
    def __init__(self, mcp_client: MCPClient, use_react: bool = True):
        super().__init__(mcp_client)
        self.use_react = use_react
        self.react_agent = ReActAgent(mcp_client) if use_react else None
        self.last_execution_trace = None
    
    def process_query(self, query: str, use_react: bool = None) -> str:
        """Process a query using either ReAct agent or traditional logic"""
        use_react = use_react if use_react is not None else self.use_react
        
        # Reset execution trace
        self.last_execution_trace = None
        
        if use_react and self.react_agent and self.react_agent.azure_client:
            result = self.react_agent.reason_and_act(query)
            self.last_execution_trace = self.react_agent.get_execution_trace()
            return result
        else:
            # Fall back to traditional processing
            return super().process_query(query)
    
    def process_query_with_trace(self, query: str, use_react: bool = None) -> Tuple[str, Optional[List[Dict]]]:
        """Process a query and return both result and execution trace"""
        result = self.process_query(query, use_react)
        return result, self.last_execution_trace
    
    def get_last_execution_trace(self) -> Optional[List[Dict]]:
        """Get the execution trace from the last query"""
        return self.last_execution_trace 