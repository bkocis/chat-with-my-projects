#!/usr/bin/env python3
"""
MCP HTTP Client for Credit Card Server

This client connects to the credit card MCP server and provides
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
    """HTTP-based MCP client for the credit card server"""
    
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
                            "name": "streamlit-credit-card-client",
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
        """Get debug status from the server"""
        return self.call_tool("debug_data_status", {})
    
    def search_credit_cards(self, search_term: str = "", card_type: str = None, 
                          max_annual_cost: float = None) -> str:
        """Search for credit cards"""
        args = {}
        if search_term:
            args["search_term"] = search_term
        if card_type:
            args["card_type"] = card_type
        if max_annual_cost is not None:
            args["max_annual_cost"] = max_annual_cost
            
        return self.call_tool("search_credit_cards", args)
    
    def search_cards_by_bank(self, bank_name: str) -> str:
        """Search for credit cards by bank name"""
        return self.call_tool("search_cards_by_bank", {"bank_name": bank_name})
    
    def find_best_cards_for_intent(self, user_intent: str, budget: float = None, 
                                 required_features: List[str] = None) -> str:
        """Find best cards for user intent"""
        args = {"user_intent": user_intent}
        if budget is not None:
            args["budget"] = budget
        if required_features:
            args["required_features"] = required_features
            
        return self.call_tool("find_best_cards_for_intent", args)
    
    def compare_credit_cards(self, product_ids: List[int], 
                           criteria: List[str] = None) -> str:
        """Compare multiple credit cards"""
        args = {"product_ids": product_ids}
        if criteria:
            args["comparison_criteria"] = criteria
            
        return self.call_tool("compare_credit_cards", args)
    
    def sql_query(self, query: str) -> str:
        """Execute SQL query"""
        return self.call_tool("sql_query", {"query": query})


class QueryProcessor:
    """Process natural language queries and determine appropriate MCP tool calls"""
    
    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
    
    def process_query(self, query: str) -> str:
        """Process a natural language query and return appropriate response"""
        query_lower = query.lower()
        
        # Debug/status queries
        if any(word in query_lower for word in ["status", "debug", "data", "loaded", "available"]):
            return self.client.get_debug_status()
        
        # Bank-specific searches
        if "bank" in query_lower:
            # Extract bank name
            words = query.split()
            bank_idx = next((i for i, word in enumerate(words) if word.lower() == "bank"), -1)
            if bank_idx >= 0 and bank_idx < len(words) - 1:
                bank_name = words[bank_idx + 1]
                return self.client.search_cards_by_bank(bank_name)
        
        # Intent-based searches
        if any(word in query_lower for word in ["best", "recommend", "suitable", "intent", "need", "want"]):
            # Extract budget if mentioned
            budget = self._extract_budget(query)
            features = self._extract_features(query)
            return self.client.find_best_cards_for_intent(query, budget, features)
        
        # Comparison queries
        if any(word in query_lower for word in ["compare", "comparison", "versus", "vs", "difference"]):
            # Try to extract product IDs or suggest using search first
            product_ids = self._extract_product_ids(query)
            if product_ids and len(product_ids) >= 2:
                return self.client.compare_credit_cards(product_ids)
            else:
                return "To compare credit cards, please first search for cards and provide their IDs. Example: 'compare cards 1, 2, 3'"
        
        # SQL queries
        if query_lower.startswith("sql:") or "select" in query_lower:
            sql_query = query.replace("sql:", "").strip()
            return self.client.sql_query(sql_query)
        
        # General search
        card_type = self._extract_card_type(query)
        max_cost = self._extract_budget(query)
        
        return self.client.search_credit_cards(query, card_type, max_cost)
    
    def _extract_budget(self, query: str) -> Optional[float]:
        """Extract budget/cost information from query"""
        import re
        
        # Look for patterns like "under 50", "max 100", "budget 75", "€50", "$50"
        patterns = [
            r"under\s+(\d+)",
            r"max\s+(\d+)",
            r"budget\s+(\d+)",
            r"[€$]\s*(\d+)",
            r"(\d+)\s*[€$]",
            r"cost\s+(\d+)",
            r"below\s+(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    def _extract_card_type(self, query: str) -> Optional[str]:
        """Extract card type from query"""
        query_lower = query.lower()
        
        if "credit" in query_lower:
            return "CREDIT"
        elif "debit" in query_lower:
            return "DEBIT"
        elif "charge" in query_lower:
            return "CHARGE"
        elif "prepaid" in query_lower:
            return "PREPAID"
        
        return None
    
    def _extract_features(self, query: str) -> List[str]:
        """Extract required features from query"""
        features = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["travel", "insurance", "abroad"]):
            features.append("travel_insurance")
        if any(word in query_lower for word in ["mobile", "contactless", "apple pay", "google pay"]):
            features.append("mobile_payment")
        if "worldwide" in query_lower and "payment" in query_lower:
            features.append("free_worldwide_payments")
        if "worldwide" in query_lower and "withdrawal" in query_lower:
            features.append("free_worldwide_withdrawal")
        
        return features
    
    def _extract_product_ids(self, query: str) -> List[int]:
        """Extract product IDs from query"""
        import re
        
        # Look for patterns like "1, 2, 3" or "cards 1 2 3" or "IDs 1,2,3"
        numbers = re.findall(r'\b\d+\b', query)
        try:
            return [int(n) for n in numbers if 1 <= int(n) <= 1000]
        except ValueError:
            return []



class ReActAgent:
    """
    Reasoning and Acting agent using Azure OpenAI GPT-4o model.
    Implements the ReAct pattern for intelligent tool usage.
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.azure_client = None
        self.conversation_context = []  # Store execution trace
        self._initialize_azure_client()
        
        # Available tools description for the o1 model
        self.tools_description = {
            "search_credit_cards": "Search for credit cards with optional filters (search_term, card_type, max_annual_cost)",
            "search_cards_by_bank": "Find credit cards from a specific bank (bank_name)",
            "find_best_cards_for_intent": "Find cards matching user intent (user_intent, budget, required_features)",
            "compare_credit_cards": "Compare multiple cards by IDs (product_ids, comparison_criteria)",
            "get_detailed_card_info": "Get comprehensive details about a specific credit card (product_id)",
            "analyze_user_preferences": "Analyze user profile and provide personalized recommendations (age_group, spending_habits, usage_pattern)",
            "sql_query": "Execute SQL queries on the credit card database (query)",
            "debug_data_status": "Get debug information about the server data status"
        }
    
    def _initialize_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            self.azure_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-0513-eu")
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
You are an intelligent credit card assistant that helps users find and compare credit cards. 
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
1. What information the user is asking for
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
            if tool_name == "search_credit_cards":
                return self.mcp_client.search_credit_cards(
                    search_term=args.get("search_term", ""),
                    card_type=args.get("card_type"),
                    max_annual_cost=args.get("max_annual_cost")
                )
            elif tool_name == "search_cards_by_bank":
                return self.mcp_client.search_cards_by_bank(args.get("bank_name", ""))
            elif tool_name == "find_best_cards_for_intent":
                return self.mcp_client.find_best_cards_for_intent(
                    user_intent=args.get("user_intent", ""),
                    budget=args.get("budget"),
                    required_features=args.get("required_features")
                )
            elif tool_name == "compare_credit_cards":
                return self.mcp_client.compare_credit_cards(
                    product_ids=args.get("product_ids", []),
                    criteria=args.get("comparison_criteria")
                )
            elif tool_name == "get_detailed_card_info":
                return self.mcp_client.call_tool("get_detailed_card_info", {
                    "product_id": args.get("product_id")
                })
            elif tool_name == "analyze_user_preferences":
                return self.mcp_client.call_tool("analyze_user_preferences", {
                    "age_group": args.get("age_group"),
                    "spending_habits": args.get("spending_habits"),
                    "usage_pattern": args.get("usage_pattern")
                })
            elif tool_name == "sql_query":
                return self.mcp_client.sql_query(args.get("query", ""))
            elif tool_name == "debug_data_status":
                return self.mcp_client.get_debug_status()
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
Based on the user query and the information gathered from the credit card database, provide a comprehensive and helpful answer.

## User Query:
{user_query}

## Information Gathered:
{context_summary}

Provide a clear, well-formatted response that directly answers the user's question. Include specific details about credit cards, their features, costs, and benefits where relevant.
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
    Implements the ReAct pattern for intelligent tool usage with local LLMs.
    """
    
    def __init__(self, mcp_client: MCPClient, model_name: str = None, ollama_host: str = None):
        self.mcp_client = mcp_client
        self.model_name = model_name or os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_available = OLLAMA_AVAILABLE
        self.conversation_context = []  # Store execution trace
        
        # Available tools description for the model
        self.tools_description = {
            "search_credit_cards": "Search for credit cards with optional filters (search_term, card_type, max_annual_cost)",
            "search_cards_by_bank": "Find credit cards from a specific bank (bank_name)",
            "find_best_cards_for_intent": "Find cards matching user intent (user_intent, budget, required_features)",
            "compare_credit_cards": "Compare multiple cards by IDs (product_ids, comparison_criteria)",
            "get_detailed_card_info": "Get comprehensive details about a specific credit card (product_id)",
            "analyze_user_preferences": "Analyze user profile and provide personalized recommendations (age_group, spending_habits, usage_pattern)",
            "sql_query": "Execute SQL queries on the credit card database (query)",
            "debug_data_status": "Get debug information about the server data status"
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
        
        prompt = f"""You are an intelligent credit card assistant that helps users find and compare credit cards. 
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
1. What information the user is asking for
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
            if tool_name == "search_credit_cards":
                return self.mcp_client.search_credit_cards(
                    search_term=args.get("search_term", ""),
                    card_type=args.get("card_type"),
                    max_annual_cost=args.get("max_annual_cost")
                )
            elif tool_name == "search_cards_by_bank":
                return self.mcp_client.search_cards_by_bank(args.get("bank_name", ""))
            elif tool_name == "find_best_cards_for_intent":
                return self.mcp_client.find_best_cards_for_intent(
                    user_intent=args.get("user_intent", ""),
                    budget=args.get("budget"),
                    required_features=args.get("required_features")
                )
            elif tool_name == "compare_credit_cards":
                return self.mcp_client.compare_credit_cards(
                    product_ids=args.get("product_ids", []),
                    criteria=args.get("comparison_criteria")
                )
            elif tool_name == "get_detailed_card_info":
                return self.mcp_client.call_tool("get_detailed_card_info", {
                    "product_id": args.get("product_id")
                })
            elif tool_name == "analyze_user_preferences":
                return self.mcp_client.call_tool("analyze_user_preferences", {
                    "age_group": args.get("age_group"),
                    "spending_habits": args.get("spending_habits"),
                    "usage_pattern": args.get("usage_pattern")
                })
            elif tool_name == "sql_query":
                return self.mcp_client.sql_query(args.get("query", ""))
            elif tool_name == "debug_data_status":
                return self.mcp_client.get_debug_status()
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
            
            final_prompt = f"""Based on the user query and the information gathered from the credit card database, provide a comprehensive and helpful answer.

## User Query:
{user_query}

## Information Gathered:
{context_summary}

Provide a clear, well-formatted response that directly answers the user's question. Include specific details about credit cards, their features, costs, and benefits where relevant.
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
                print(f"Initialized Ollama ReAct agent with model: {self.react_agent.model_name}")
            else:  # default to azure
                self.react_agent = ReActAgent(self.client)
                print("Initialized Azure OpenAI ReAct agent")
        except Exception as e:
            print(f"Failed to initialize {self.agent_type} ReAct agent: {e}")
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
        
        if use_react and self.react_agent:
            # Check if the agent is properly initialized
            if isinstance(self.react_agent, OllamaReActAgent):
                if not self.react_agent.ollama_available:
                    print("Ollama agent not available, falling back to traditional processing")
                    return super().process_query(query)
            elif isinstance(self.react_agent, ReActAgent):
                if not self.react_agent.azure_client:
                    print("Azure OpenAI agent not available, falling back to traditional processing")
                    return super().process_query(query)
            
            # Use ReAct agent
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