#!/usr/bin/env python3
"""
GitHub Repositories MCP HTTP Server

An HTTP-based MCP server that provides tools to explore GitHub repositories.
Runs on localhost:8081 for easy integration.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# --- MCP server stubs (replace with your actual MCP server framework if needed) ---
class DummyTypes:
    class Resource:
        def __init__(self, uri, name, description, mimeType):
            self.uri = uri
            self.name = name
            self.description = description
            self.mimeType = mimeType
        def model_dump(self, exclude_none=True):
            return self.__dict__
    class Tool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
        def model_dump(self, exclude_none=True):
            return self.__dict__
    class TextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text
        def model_dump(self):
            return self.__dict__
    class AnyUrl(str):
        pass

types = DummyTypes()

# --- Configurable root directory ---
GITHUB_ROOT = os.environ.get(
    "GITHUB_REPOS_ROOT",
    "/home/snow/Documents/Projects/github-repositories/bkocis"
)

# --- Session management ---
SESSIONS = {}

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
async def handle_list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="list_readme_files",
            description="List all README files in all repositories.",
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
            description="Get the content of all README files as a list of {path, content} objects.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="get_all_readmes_summary",
            description="Get a summary (first 10 lines or 1000 chars) of all README files as a list of {path, summary} objects.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
    ]

async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
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

async def handle_root(request: Request):
    # Return a simple MCP-compatible info response
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "name": "GitHub Repos MCP Server",
            "description": "Explore and process README files in local GitHub repositories.",
            "mcp_version": "0.1.0",
            "endpoints": ["/mcp"]
        }
    })

# --- HTTP endpoint handlers ---
async def handle_mcp_endpoint(request: Request):
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"}
            }, status_code=400)
        method = body.get("method")
        params = body.get("params", {})
        msg_id = body.get("id")
        # --- MCP session and initialize handling ---
        session_id = request.headers.get("mcp-session-id")
        import uuid
        if not session_id and method == "initialize":
            session_id = str(uuid.uuid4())
            SESSIONS[session_id] = {}
        elif session_id and session_id not in SESSIONS and method != "initialize":
            return JSONResponse({"error": "Session not found"}, status_code=404)
        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
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
            headers = {"mcp-session-id": session_id} if session_id else {}
            return JSONResponse(response, headers=headers)
        if not method:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32600, "message": "Invalid Request: 'method' is required"}
            }, status_code=400)
        try:
            if method == "initialize":
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "name": "GitHub Repos MCP Server",
                        "description": "Explore and process README files in local GitHub repositories.",
                        "mcp_version": "0.1.0",
                        "capabilities": {
                            "tools": True
                        }
                    }
                })
            elif method == "tools/list":
                tools = await handle_list_tools()
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": [tool.model_dump() for tool in tools]}
                })
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                result = await handle_call_tool(name, arguments)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [item.model_dump() for item in result]}
                })
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }, status_code=400)
        except Exception as e:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)}
            }, status_code=500)
    else:
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

# --- Main server setup ---
def main():
    app = Starlette(routes=[
        Route("/", handle_root, methods=["GET"]),
        Route("/mcp", handle_mcp_endpoint, methods=["POST"]),
    ])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print(f"Starting GitHub Repos MCP HTTP Server on 127.0.0.1:8081")
    print(f"Root directory: {GITHUB_ROOT}")
    uvicorn.run(app, host="127.0.0.1", port=8081)

if __name__ == "__main__":
    main() 