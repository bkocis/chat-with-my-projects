#!/usr/bin/env python3
"""
GitHub Repository Explorer - Streamlit App

A natural language interface for exploring GitHub repositories
using the MCP (Model Context Protocol) server.
"""

import streamlit as st
import time
import json
from mcp_client import MCPClient, QueryProcessor, EnhancedQueryProcessor


# Page configuration
st.set_page_config(
    page_title="GitHub Repository Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .query-box {
        margin: 2rem 0;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .example-query {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        display: inline-block;
        margin-left: 1rem;
    }
    .status-connected {
        background-color: #d4edda;
        color: #155724;
    }
    .status-disconnected {
        background-color: #f8d7da;
        color: #721c24;
    }
    .reasoning-step {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .tool-action {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
    }
    .tool-result {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
    }
    .iteration-header {
        color: #6f42c1;
        font-weight: bold;
        border-bottom: 2px solid #6f42c1;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .repo-file {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .readme-content {
        background-color: #fff;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_mcp_client():
    """Get or create MCP client (cached)"""
    return MCPClient()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_server_status(_client):
    """Get server status with caching"""
    try:
        tools = _client.list_tools()
        return True, f"Available tools: {[tool.get('name', 'Unknown') for tool in tools]}"
    except Exception as e:
        return False, str(e)


def display_tool_execution_trace(execution_trace):
    """Display the tool execution trace in an organized manner"""
    if not execution_trace:
        return
    
    st.subheader("🔍 AI Reasoning & Tool Execution Trace")
    
    # Overall summary
    total_iterations = len(execution_trace)
    total_actions = sum(len(ctx.get('actions', [])) for ctx in execution_trace)
    
    st.info(f"**Execution Summary:** {total_iterations} reasoning iterations, {total_actions} tool calls")
    
    # Show each iteration
    for ctx in execution_trace:
        iteration = ctx.get('iteration', 0)
        reasoning = ctx.get('reasoning', '')
        actions = ctx.get('actions', [])
        results = ctx.get('results', [])
        
        with st.expander(f"🧠 Iteration {iteration} - AI Reasoning & Actions", expanded=(iteration <= 2)):
            # Show reasoning
            if reasoning:
                st.markdown('<div class="iteration-header">🤔 AI Reasoning:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="reasoning-step">{reasoning}</div>', unsafe_allow_html=True)
            
            # Show actions and results
            if actions:
                st.markdown('<div class="iteration-header">⚙️ Tool Actions & Results:</div>', unsafe_allow_html=True)
                
                for i, action_result in enumerate(results):
                    action = action_result.get('action', {})
                    result = action_result.get('result', '')
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f'<div class="tool-action"><strong>🔧 Tool Call {i+1}:</strong><br>'
                                  f'<strong>Tool:</strong> {action.get("tool", "Unknown")}<br>'
                                  f'<strong>Arguments:</strong><br><code>{json.dumps(action.get("args", {}), indent=2)}</code></div>', 
                                  unsafe_allow_html=True)
                    
                    with col2:
                        # Truncate very long results for display
                        display_result = result[:1000] + "..." if len(result) > 1000 else result
                        st.markdown(f'<div class="tool-result"><strong>📊 Result:</strong><br><pre>{display_result}</pre></div>', 
                                  unsafe_allow_html=True)
                        
                        # Option to show full result
                        if len(result) > 1000:
                            if st.button(f"Show Full Result {i+1}", key=f"full_result_{iteration}_{i}"):
                                st.text_area("Full Result:", value=result, height=200, key=f"full_text_{iteration}_{i}")


def display_query_metrics(start_time, end_time, execution_trace=None):
    """Display query execution metrics"""
    execution_time = end_time - start_time
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Execution Time", f"{execution_time:.2f}s")
    
    with col2:
        iterations = len(execution_trace) if execution_trace else 1
        st.metric("🔄 AI Iterations", iterations)
    
    with col3:
        total_actions = sum(len(ctx.get('actions', [])) for ctx in execution_trace) if execution_trace else 0
        st.metric("🔧 Tool Calls", total_actions)
    
    with col4:
        mode = "🤖 ReAct AI" if execution_trace else "📝 Traditional"
        st.metric("🧠 Mode", mode)


def display_readme_files(files_list):
    """Display README files in a nice format"""
    if not files_list:
        st.warning("No README files found")
        return
    
    try:
        files = json.loads(files_list) if isinstance(files_list, str) else files_list
        st.write(f"**Found {len(files)} README files:**")
        
        for file_path in files:
            st.markdown(f'<div class="repo-file">📄 {file_path}</div>', unsafe_allow_html=True)
    except:
        st.text(files_list)


def display_readme_content(content, title="README Content"):
    """Display README content in a formatted way"""
    st.subheader(title)
    
    try:
        if isinstance(content, str) and content.startswith('['):
            # It's a JSON list of content objects
            content_list = json.loads(content)
            for item in content_list:
                path = item.get('path', 'Unknown')
                text_content = item.get('content', '') or item.get('summary', '')
                
                with st.expander(f"📄 {path.split('/')[-2:]}", expanded=False):
                    st.markdown(f'<div class="readme-content">{text_content}</div>', unsafe_allow_html=True)
        else:
            # It's a single content string
            st.markdown(f'<div class="readme-content">{content}</div>', unsafe_allow_html=True)
    except:
        st.text(content)


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 GitHub Repository Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Explore and analyze GitHub repositories with AI-powered insights</h3>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = get_mcp_client()
        st.session_state.processor = EnhancedQueryProcessor(st.session_state.mcp_client)
        st.session_state.use_react = True
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings & Status")
        
        # Server connection status
        st.subheader("Server Status")
        
        if st.button("🔄 Check Connection"):
            st.session_state.connection_status = None  # Reset cache
        
        try:
            is_connected, status_info = get_server_status(st.session_state.mcp_client)
            if is_connected:
                st.markdown('<span class="status-indicator status-connected">🟢 Connected</span>', 
                          unsafe_allow_html=True)
                with st.expander("📊 Available Tools"):
                    st.text(status_info)
            else:
                st.markdown('<span class="status-indicator status-disconnected">🔴 Disconnected</span>', 
                          unsafe_allow_html=True)
                st.error(f"Connection failed: {status_info}")
                st.info("Make sure the MCP server is running on localhost:8081")
        except Exception as e:
            st.markdown('<span class="status-indicator status-disconnected">🔴 Error</span>', 
                      unsafe_allow_html=True)
            st.error(f"Error: {e}")
        
        st.divider()
        
        # ReAct Mode Toggle
        st.subheader("🧠 AI Mode")
        use_react = st.toggle(
            "🤖 Use ReAct AI Agent", 
            value=st.session_state.get('use_react', True),
            help="Enable for intelligent reasoning with AI model"
        )
        
        if use_react != st.session_state.get('use_react', True):
            st.session_state.use_react = use_react
            st.rerun()
        
        # Show current mode status
        if use_react:
            # Check if Azure OpenAI is actually available
            try:
                from mcp_client import ReActAgent
                test_agent = ReActAgent(st.session_state.mcp_client)
                if test_agent.azure_client:
                    st.success("🤖 AI Agent Mode: Active")
                    st.caption("Using Azure OpenAI for intelligent reasoning")
                else:
                    st.warning("🤖 AI Agent Mode: Unavailable")
                    st.caption("Missing Azure OpenAI environment variables")
                    st.caption("Will fall back to traditional mode")
            except Exception:
                st.warning("🤖 AI Agent Mode: Error")
                st.caption("Azure OpenAI configuration issue")
        else:
            st.info("📝 Traditional Mode: Active")
            st.caption("Using rule-based query processing")
        
        # Repository Configuration
        st.divider()
        st.subheader("📁 Repository Configuration")
        
        import os
        
        with st.expander("🔗 Server Configuration", expanded=False):
            mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8081/mcp")
            repo_root = os.getenv("GITHUB_REPOS_ROOT", "/home/snow/Documents/Projects/github-repositories/bkocis")
            
            st.info(f"🌐 Server URL: {mcp_url}")
            st.info(f"📂 Repository Root: {repo_root}")
            
            # Test MCP connection
            if st.button("🧪 Test MCP Server Connection", key="test_mcp"):
                with st.spinner("Testing MCP connection..."):
                    try:
                        if st.session_state.mcp_client.initialize():
                            st.success("🎉 MCP Server connection successful!")
                            st.caption(f"Session ID: {st.session_state.mcp_client.session_id}")
                        else:
                            st.error("❌ Failed to connect to MCP server")
                    except Exception as e:
                        st.error(f"❌ MCP connection test failed: {str(e)}")
        
        # Display options
        st.divider()
        st.subheader("📊 Display Options")
        
        show_trace = st.checkbox(
            "🔍 Show Tool Execution Trace", 
            value=st.session_state.get('show_trace', True),
            help="Display AI reasoning steps and tool calls"
        )
        st.session_state.show_trace = show_trace
        
        show_metrics = st.checkbox(
            "📈 Show Execution Metrics", 
            value=st.session_state.get('show_metrics', True),
            help="Display timing and performance metrics"
        )
        st.session_state.show_metrics = show_metrics
        
        st.divider()
        
        # Query examples
        st.subheader("💡 Example Queries")
        
        examples = [
            "List all README files in the repositories",
            "Show me a summary of all README files",
            "Get the full content of all README files",
            "Find repositories related to machine learning",
            "What projects are documented in these repositories?",
            "Show me the README content for a specific repository",
            "Summarize the technologies used across all projects",
            "What are the main purposes of these repositories?",
            "Find repositories that mention Python or JavaScript",
            "Compare the documentation quality across repositories"
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                st.session_state.current_query = example
        
        st.divider()
        
        # Query history
        st.subheader("📜 Query History")
        if st.session_state.query_history:
            for i, history_item in enumerate(reversed(st.session_state.query_history[-5:])):
                query = history_item[0] if isinstance(history_item, tuple) else history_item.get('query', '')
                if st.button(f"🔄 {query[:30]}...", key=f"history_{i}"):
                    st.session_state.current_query = query
        else:
            st.write("No queries yet")
        
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
    
    # Main query interface
    st.subheader("🔍 Ask About Your Repositories")
    
    # Query input
    query = st.text_area(
        "Enter your repository exploration question:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="e.g., 'What kind of projects are in these repositories?' or 'Show me all README files'",
        key="query_input"
    )
    
    # Query options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Explore", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Analyzing repositories..."):
                    try:
                        # Process the query with current mode
                        start_time = time.time()
                        use_react_mode = st.session_state.get('use_react', True)
                        
                        # Process query and get execution trace
                        result, execution_trace = st.session_state.processor.process_query_with_trace(query, use_react=use_react_mode)
                        
                        end_time = time.time()
                        
                        # Store in history with trace information
                        history_item = {
                            'query': query,
                            'result': result,
                            'execution_trace': execution_trace,
                            'timestamp': time.time(),
                            'execution_time': end_time - start_time,
                            'mode': 'ReAct' if (use_react_mode and execution_trace) else 'Traditional'
                        }
                        st.session_state.query_history.append(history_item)
                        
                        # Display results
                        st.subheader("📋 Results")
                        
                        # Format results based on content type
                        if "README files" in query and "list" in query.lower():
                            display_readme_files(result)
                        elif any(word in query.lower() for word in ["content", "summary", "show"]):
                            display_readme_content(result)
                        else:
                            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                        
                        # Display execution metrics if enabled
                        if st.session_state.get('show_metrics', True):
                            display_query_metrics(start_time, end_time, execution_trace)
                        
                        # Display tool execution trace if enabled and available
                        if st.session_state.get('show_trace', True) and execution_trace:
                            display_tool_execution_trace(execution_trace)
                        
                        # Clear the current query
                        if 'current_query' in st.session_state:
                            del st.session_state.current_query
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        import traceback
                        with st.expander("🐛 Error Details"):
                            st.code(traceback.format_exc())
            else:
                st.warning("Please enter a query!")
    
    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.current_query = ""
            st.rerun()
    
    with col3:
        if st.button("📊 Debug", use_container_width=True):
            with st.spinner("Getting available tools..."):
                tools = st.session_state.mcp_client.list_tools()
                st.json(tools)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        st.write("**Direct Tool Access:**")
        
        tool_col1, tool_col2 = st.columns(2)
        
        with tool_col1:
            st.write("**Repository Tools:**")
            if st.button("📄 List All README Files"):
                result = st.session_state.mcp_client.list_readme_files()
                display_readme_files(result)
            
            if st.button("📝 Get All README Summaries"):
                result = st.session_state.mcp_client.get_all_readmes_summary()
                display_readme_content(result, "README Summaries")
        
        with tool_col2:
            st.write("**Content Access:**")
            if st.button("📖 Get All README Content"):
                result = st.session_state.mcp_client.get_all_readmes_content()
                display_readme_content(result, "All README Content")
            
            # Single file content
            readme_path = st.text_input("README file path:", placeholder="e.g., /path/to/README.md")
            if st.button("Get Specific README") and readme_path:
                result = st.session_state.mcp_client.get_readme_content(readme_path)
                st.text_area("Content:", value=result, height=300)
    
    # Query History Viewer
    if st.session_state.query_history:
        st.divider()
        st.subheader("📜 Recent Query History")
        
        # Show recent queries with their traces
        for i, history_item in enumerate(reversed(st.session_state.query_history[-3:])):
            if isinstance(history_item, dict):
                query = history_item.get('query', '')
                result = history_item.get('result', '')
                execution_trace = history_item.get('execution_trace')
                mode = history_item.get('mode', 'Unknown')
                execution_time = history_item.get('execution_time', 0)
                
                with st.expander(f"🔍 Query {len(st.session_state.query_history) - i}: {query[:50]}... ({mode} mode, {execution_time:.2f}s)"):
                    st.markdown("**Query:**")
                    st.info(query)
                    
                    st.markdown("**Result:**")
                    if "README files" in query and "list" in query.lower():
                        display_readme_files(result)
                    elif any(word in query.lower() for word in ["content", "summary", "show"]):
                        display_readme_content(result)
                    else:
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                    
                    if execution_trace and st.session_state.get('show_trace', True):
                        display_tool_execution_trace(execution_trace)
    
    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #666;">
            🚀 GitHub Repository Explorer | Powered by MCP (Model Context Protocol)
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 