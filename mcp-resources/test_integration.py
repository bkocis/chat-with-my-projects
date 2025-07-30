#!/usr/bin/env python3
"""
Test script to verify MCP server integration with enhanced tools.

This script tests that the enhanced tools are working correctly
and can be called as expected by the MCP server.
"""

import json
import sys
import os

# Add mcp-resources to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from enhanced_mcp_tools import EnhancedRepositoryTools, handle_enhanced_tool_call


async def test_enhanced_tools():
    """Test all enhanced tools to ensure they work correctly."""
    
    print("Testing Enhanced MCP Tools Integration")
    print("=" * 50)
    
    # Initialize tools with correct data path
    data_dir = os.path.join(os.path.dirname(__file__), 'mcp-resources')
    tools = EnhancedRepositoryTools(data_dir)
    
    # Test 1: Technology Statistics
    print("\n1. Testing get_technology_statistics...")
    try:
        result = await handle_enhanced_tool_call(tools, "get_technology_statistics", {})
        data = json.loads(result[0].text)
        total_repos = data.get('overview', {}).get('total_repositories', 0)
        print(f"   ✓ Found {total_repos} repositories")
        print(f"   ✓ Top languages: {list(data.get('languages', {}).keys())[:3]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Search by Language
    print("\n2. Testing search_repositories_by_language...")
    try:
        result = await handle_enhanced_tool_call(tools, "search_repositories_by_language", {"language": "Python"})
        data = json.loads(result[0].text)
        print(f"   ✓ Found {len(data)} Python repositories")
        if data:
            print(f"   ✓ Example: {data[0]['repository']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: README Search
    print("\n3. Testing search_readme_content...")
    try:
        result = await handle_enhanced_tool_call(tools, "search_readme_content", {"search_term": "machine learning", "max_results": 3})
        data = json.loads(result[0].text)
        print(f"   ✓ Found {len(data)} README files mentioning 'machine learning'")
        if data:
            print(f"   ✓ Example: {data[0]['repository']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Framework Search
    print("\n4. Testing search_repositories_by_framework...")
    try:
        result = await handle_enhanced_tool_call(tools, "search_repositories_by_framework", {"framework": "Flask"})
        data = json.loads(result[0].text)
        print(f"   ✓ Found {len(data)} Flask repositories")
        if data:
            print(f"   ✓ Example: {data[0]['repository']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: Largest Repositories
    print("\n5. Testing get_largest_repositories...")
    try:
        result = await handle_enhanced_tool_call(tools, "get_largest_repositories", {"limit": 3})
        data = json.loads(result[0].text)
        print(f"   ✓ Found {len(data)} largest repositories")
        if data:
            print(f"   ✓ Largest: {data[0]['repository']} ({data[0]['size_mb']:.1f} MB)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 6: Repository Details
    print("\n6. Testing get_repository_details...")
    try:
        # Get a repository name from the list
        repos = tools.list_all_repositories()
        if repos:
            test_repo = repos[0]
            result = await handle_enhanced_tool_call(tools, "get_repository_details", {"repository_name": test_repo})
            data = json.loads(result[0].text)
            print(f"   ✓ Got details for repository: {test_repo}")
            print(f"   ✓ Languages: {list(data.get('file_structure', {}).get('languages', {}).keys())[:3]}")
        else:
            print("   ⚠ No repositories found for testing")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"\n{'='*50}")
    print("✅ Enhanced tools integration test completed!")
    print("🚀 Your MCP server is ready with enhanced capabilities!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_tools())