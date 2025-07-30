#!/usr/bin/env python3
"""
GitHub Repositories MCP HTTP Server

An HTTP-based MCP server that provides tools to explore GitHub repositories.
Uses Streamable HTTP transport protocol according to MCP specification.
Runs on localhost:8081 for easy integration.
"""

import os
import json
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server

# Import enhanced tools for preprocessed data
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mcp-resources'))
try:
    from enhanced_mcp_tools import EnhancedRepositoryTools, get_enhanced_mcp_tools, handle_enhanced_tool_call
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    ENHANCED_TOOLS_AVAILABLE = False
    print("Warning: Enhanced tools not available. Using basic filesystem tools only.")

# HTTP server dependencies
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

# Initialize the MCP server
server = Server("github-repos")

# --- Configurable root directory ---
GITHUB_ROOT = os.environ.get(
    "GITHUB_REPOS_ROOT",
    "/home/snow/Documents/Projects/github-repositories/bkocis"
)

# --- Session management ---
SESSIONS = {}

# --- Enhanced tools initialization ---
if ENHANCED_TOOLS_AVAILABLE:
    # Data is in mcp-resources/mcp-resources (nested structure from preprocessing)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'mcp-resources', 'mcp-resources')
    enhanced_tools = EnhancedRepositoryTools(data_dir)
    enhanced_tool_list = get_enhanced_mcp_tools()
else:
    enhanced_tools = None
    enhanced_tool_list = []

# --- Tool implementations ---
def find_all_readme_files() -> List[str]:
    """Recursively find all README files under the root directory."""
    readme_files = []
    for dirpath, dirnames, filenames in os.walk(GITHUB_ROOT):
        for fname in filenames:
            if fname.lower().startswith("readme"):
                readme_files.append(os.path.join(dirpath, fname))
    return readme_files

def get_readme_content(path: str) -> str:
    """Read and return the content of a README file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

# --- MCP tool handlers ---
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    # Basic filesystem tools (always available)
    basic_tools = [
        types.Tool(
            name="list_readme_files",
            description="List all README files in all repositories (filesystem scan).",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="get_readme_content",
            description="Get the content of a specific README file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the README file (absolute or relative to root)"}
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="get_all_readmes_content",
            description="Get the content of all README files as a list of {path, content} objects (filesystem scan).",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="get_all_readmes_summary",
            description="Get a summary (first 10 lines or 1000 chars) of all README files as a list of {path, summary} objects (filesystem scan).",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
    ]
    
    # Add enhanced tools if available
    if ENHANCED_TOOLS_AVAILABLE and enhanced_tool_list:
        return basic_tools + enhanced_tool_list
    else:
        return basic_tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    # Check if this is an enhanced tool
    if ENHANCED_TOOLS_AVAILABLE and enhanced_tools:
        enhanced_tool_names = [tool.name for tool in enhanced_tool_list]
        if name in enhanced_tool_names:
            return await handle_enhanced_tool_call(enhanced_tools, name, arguments)
    
    # Handle basic filesystem tools
    if name == "list_readme_files":
        files = find_all_readme_files()
        return [types.TextContent(type="text", text=json.dumps(files, indent=2))]
    elif name == "get_readme_content":
        path = arguments["path"]
        # If relative, make absolute
        if not os.path.isabs(path):
            path = os.path.join(GITHUB_ROOT, path)
        content = get_readme_content(path)
        return [types.TextContent(type="text", text=content)]
    elif name == "get_all_readmes_content":
        files = find_all_readme_files()
        results = []
        for path in files:
            content = get_readme_content(path)
            results.append({"path": path, "content": content})
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
    elif name == "get_all_readmes_summary":
        files = find_all_readme_files()
        results = []
        for path in files:
            content = get_readme_content(path)
            # Summary: first 10 lines or 1000 chars, whichever is shorter
            lines = content.splitlines()
            summary = "\n".join(lines[:10])
            if len(summary) > 1000:
                summary = summary[:1000] + "..."
            results.append({"path": path, "summary": summary})
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

class HTTPTransport:
    """Custom HTTP transport for MCP following Streamable HTTP specification"""
    
    def __init__(self):
        self.read_queue = asyncio.Queue()
        self.write_queue = asyncio.Queue()
        self.session_id = None
        
    async def read_stream(self):
        """Read messages from the transport"""
        while True:
            message = await self.read_queue.get()
            if message is None:  # Shutdown signal
                break
            yield message
    
    async def write_stream(self, message):
        """Write messages to the transport"""
        await self.write_queue.put(message)


# HTTP endpoint handlers
async def handle_mcp_endpoint(request: Request):
    """Handle MCP endpoint requests (both GET and POST)"""
    
    # Security: Validate Origin header to prevent DNS rebinding attacks
    # More permissive in Docker environment
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
    origin = request.headers.get("origin")
    
    if not is_docker and origin and origin not in ["http://localhost:8081", "http://127.0.0.1:8081"]:
        return JSONResponse({"error": "Invalid origin"}, status_code=403)
    
    # Check protocol version header
    protocol_version = request.headers.get("mcp-protocol-version", "2025-03-26")
    if protocol_version not in ["2025-03-26", "2025-06-18"]:
        return JSONResponse({"error": "Unsupported protocol version"}, status_code=400)
    
    if request.method == "POST":
        return await handle_post_request(request)
    elif request.method == "GET":
        return await handle_get_request(request)
    elif request.method == "DELETE":
        return await handle_delete_request(request)
    else:
        return JSONResponse({"error": "Method not allowed"}, status_code=405)


async def handle_post_request(request: Request):
    """Handle POST requests for client-to-server messages"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    # Get or create session
    session_id = request.headers.get("mcp-session-id")
    if not session_id and body.get("method") == "initialize":
        # Create new session for initialize request
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"transport": HTTPTransport()}
    elif session_id and session_id not in SESSIONS:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    # Handle initialize request specially
    if body.get("method") == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "resources": {},
                    "tools": {},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "github-repos",
                    "version": "1.0.0"
                }
            }
        }
        
        headers = {"mcp-session-id": session_id}
        return JSONResponse(response, headers=headers)
    
    # For other requests, validate session
    if not session_id or session_id not in SESSIONS:
        return JSONResponse({"error": "Session required"}, status_code=400)
    
    transport = SESSIONS[session_id]["transport"]
    
    # Process the message through MCP server
    await transport.read_queue.put(json.dumps(body))
    
    # For notifications and responses, return 202 Accepted
    if "id" not in body:  # Notification
        return Response(status_code=202)
    
    # For requests, we need to get the response
    try:
        response = await handle_mcp_message(body)
        return JSONResponse(response)
    except Exception as e:
        error_response = {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }
        return JSONResponse(error_response, status_code=500)


async def handle_get_request(request: Request):
    """Handle GET requests for server-to-client SSE streams"""
    session_id = request.headers.get("mcp-session-id")
    if not session_id or session_id not in SESSIONS:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    # Return SSE stream for server-initiated messages
    async def event_stream():
        transport = SESSIONS[session_id]["transport"]
        while True:
            try:
                # Wait for messages from server
                message = await asyncio.wait_for(transport.write_queue.get(), timeout=30)
                yield f"data: {json.dumps(message)}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield "data: {}\n\n"
            except Exception:
                break
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


async def handle_delete_request(request: Request):
    """Handle DELETE requests for session termination"""
    session_id = request.headers.get("mcp-session-id")
    if session_id and session_id in SESSIONS:
        del SESSIONS[session_id]
        return Response(status_code=200)
    return JSONResponse({"error": "Session not found"}, status_code=404)


async def handle_mcp_message(message: dict) -> dict:
    """Process MCP messages and return responses"""
    method = message.get("method")
    params = message.get("params", {})
    msg_id = message.get("id")
    
    try:
        if method == "tools/list":
            tools = await handle_list_tools()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": [tool.model_dump(exclude_none=True) for tool in tools]}
            }
        
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            result = await handle_call_tool(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [item.model_dump() for item in result]}
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
    
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32603, "message": str(e)}
        }

def main():
    """Main HTTP server function"""
    
    # Create Starlette application
    app = Starlette(routes=[
        Route("/mcp", handle_mcp_endpoint, methods=["GET", "POST", "DELETE"]),
    ])
    
    # Detect if running in Docker container
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
    
    # Set host based on environment
    if is_docker:
        host = "0.0.0.0"  # Accept connections from outside container
        allowed_origins = ["*"]  # More permissive for container
    else:
        host = "127.0.0.1"  # Localhost only for local development
        allowed_origins = ["http://localhost:8081", "http://127.0.0.1:8081"]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    port = int(os.getenv("MCP_SERVER_PORT", "8081"))
    
    print(f"Starting GitHub Repos MCP HTTP Server on {host}:{port}")
    print(f"MCP endpoint: http://{'localhost' if host == '127.0.0.1' else host}:{port}/mcp")
    print("Protocol: Streamable HTTP Transport")
    print(f"Environment: {'Docker Container' if is_docker else 'Local Development'}")
    print(f"Root directory: {GITHUB_ROOT}")
    
    # Enhanced tools status
    if ENHANCED_TOOLS_AVAILABLE and enhanced_tools:
        print(f"✓ Enhanced tools: {len(enhanced_tool_list)} preprocessing-based tools available")
        try:
            # Show data status
            repos = enhanced_tools.list_all_repositories()
            stats = enhanced_tools.get_technology_statistics()
            total_repos = stats.get('overview', {}).get('total_repositories', 0)
            print(f"✓ Preprocessed data: {total_repos} repositories, {stats.get('overview', {}).get('total_files', 0):,} files")
        except Exception as e:
            print(f"⚠ Enhanced tools available but data not accessible: {e}")
    else:
        print("⚠ Enhanced tools not available - using basic filesystem tools only")
        print("  Run preprocessing first: cd mcp-resources && python run_preprocessing.py")
    
    # Bind based on environment
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main() 