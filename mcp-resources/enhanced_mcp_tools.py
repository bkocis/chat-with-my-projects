#!/usr/bin/env python3
"""
Enhanced MCP Tools using Preprocessed Data

This module provides enhanced MCP tools that use the preprocessed repository data
for faster queries and more sophisticated analysis capabilities.

These tools can be integrated into the main MCP server to provide:
- Fast repository queries
- Technology stack filtering
- Semantic search across README files
- Repository recommendations
- Technology trend analysis
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import mcp.types as types


class EnhancedRepositoryTools:
    """Enhanced repository analysis tools using preprocessed data."""
    
    def __init__(self, data_dir: str = 'mcp-resources'):
        self.data_dir = Path(data_dir)
        self.summary_file = self.data_dir / 'repositories_summary.json'
        self.readme_index_file = self.data_dir / 'readme_index.json'
        self.repos_dir = self.data_dir / 'repositories'
        
        # Load cached data
        self._load_cached_data()
    
    def _load_cached_data(self):
        """Load preprocessed data into memory for fast access."""
        try:
            if self.summary_file.exists():
                with open(self.summary_file, 'r') as f:
                    self.summary = json.load(f)
            else:
                self.summary = {}
            
            if self.readme_index_file.exists():
                with open(self.readme_index_file, 'r') as f:
                    self.readme_index = json.load(f)
            else:
                self.readme_index = {}
        
        except Exception as e:
            print(f"Error loading cached data: {e}")
            self.summary = {}
            self.readme_index = {}
    
    def get_repository_data(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed data for a specific repository."""
        try:
            repo_file = self.repos_dir / f"{repo_name}.json"
            if repo_file.exists():
                with open(repo_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading repository data for {repo_name}: {e}")
        return None
    
    def list_all_repositories(self) -> List[str]:
        """Get list of all available repositories."""
        return self.summary.get('repository_list', [])
    
    def search_repositories_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Find repositories that use a specific programming language."""
        results = []
        
        for repo_name in self.list_all_repositories():
            repo_data = self.get_repository_data(repo_name)
            if repo_data:
                languages = repo_data.get('file_structure', {}).get('languages', {})
                if language in languages:
                    results.append({
                        'repository': repo_name,
                        'language_file_count': languages[language],
                        'total_files': repo_data.get('file_structure', {}).get('total_files', 0),
                        'size_mb': repo_data.get('file_structure', {}).get('total_size', 0) / (1024 * 1024),
                        'technologies': repo_data.get('technologies', {})
                    })
        
        # Sort by file count of the specific language
        results.sort(key=lambda x: x['language_file_count'], reverse=True)
        return results
    
    def search_repositories_by_technology(self, tech_type: str, technology: str) -> List[Dict[str, Any]]:
        """Find repositories using specific technology (framework, tool, database, etc.)."""
        results = []
        
        for repo_name in self.list_all_repositories():
            repo_data = self.get_repository_data(repo_name)
            if repo_data:
                technologies = repo_data.get('technologies', {})
                tech_list = technologies.get(tech_type, [])
                
                if technology in tech_list:
                    results.append({
                        'repository': repo_name,
                        'technologies': technologies,
                        'languages': repo_data.get('file_structure', {}).get('languages', {}),
                        'size_mb': repo_data.get('file_structure', {}).get('total_size', 0) / (1024 * 1024),
                        'last_commit': repo_data.get('git_info', {}).get('last_commit', {})
                    })
        
        return results
    
    def search_readme_content(self, search_term: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search README files for specific content."""
        results = []
        search_term_lower = search_term.lower()
        
        for readme in self.readme_index.get('readmes', []):
            summary = readme.get('summary', '').lower()
            if search_term_lower in summary:
                # Get repository data for additional context
                repo_data = self.get_repository_data(readme['repository'])
                
                result = {
                    'repository': readme['repository'],
                    'readme_path': readme['file_path'],
                    'summary': readme['summary'],
                    'word_count': readme['word_count'],
                    'last_modified': readme['last_modified']
                }
                
                if repo_data:
                    result['technologies'] = repo_data.get('technologies', {})
                    result['languages'] = repo_data.get('file_structure', {}).get('languages', {})
                
                results.append(result)
        
        return results[:max_results]
    
    def get_technology_statistics(self) -> Dict[str, Any]:
        """Get comprehensive technology usage statistics."""
        if not self.summary:
            return {}
        
        stats = self.summary.get('summary', {})
        
        # Calculate additional derived statistics
        total_repos = self.summary.get('metadata', {}).get('total_repositories', 0)
        
        technology_stats = {
            'overview': {
                'total_repositories': total_repos,
                'total_files': stats.get('total_files', 0),
                'total_size_mb': stats.get('total_size', 0) / (1024 * 1024)
            },
            'languages': dict(Counter(stats.get('languages', {})).most_common()),
            'frameworks': dict(Counter(stats.get('frameworks', {})).most_common()),
            'tools': dict(Counter(stats.get('tools', {})).most_common())
        }
        
        # Calculate percentages
        if total_repos > 0:
            for category in ['languages', 'frameworks', 'tools']:
                for tech, count in technology_stats[category].items():
                    technology_stats[category][tech] = {
                        'count': count,
                        'percentage': round((count / total_repos) * 100, 1)
                    }
        
        return technology_stats
    
    def get_repository_recommendations(self, repo_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for similar repositories based on technology stack."""
        target_repo = self.get_repository_data(repo_name)
        if not target_repo:
            return []
        
        target_technologies = target_repo.get('technologies', {})
        target_languages = set(target_repo.get('file_structure', {}).get('languages', {}).keys())
        
        similarities = []
        
        for other_repo_name in self.list_all_repositories():
            if other_repo_name == repo_name:
                continue
            
            other_repo = self.get_repository_data(other_repo_name)
            if not other_repo:
                continue
            
            other_technologies = other_repo.get('technologies', {})
            other_languages = set(other_repo.get('file_structure', {}).get('languages', {}).keys())
            
            # Calculate similarity score
            similarity_score = 0
            
            # Language similarity
            common_languages = target_languages.intersection(other_languages)
            if target_languages and other_languages:
                language_similarity = len(common_languages) / len(target_languages.union(other_languages))
                similarity_score += language_similarity * 0.4
            
            # Framework similarity
            target_frameworks = set(target_technologies.get('frameworks', []))
            other_frameworks = set(other_technologies.get('frameworks', []))
            common_frameworks = target_frameworks.intersection(other_frameworks)
            if target_frameworks and other_frameworks:
                framework_similarity = len(common_frameworks) / len(target_frameworks.union(other_frameworks))
                similarity_score += framework_similarity * 0.6
            
            if similarity_score > 0:
                similarities.append({
                    'repository': other_repo_name,
                    'similarity_score': similarity_score,
                    'common_languages': list(common_languages),
                    'common_frameworks': list(common_frameworks),
                    'technologies': other_technologies,
                    'size_mb': other_repo.get('file_structure', {}).get('total_size', 0) / (1024 * 1024)
                })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:limit]
    
    def get_largest_repositories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the largest repositories by file size."""
        repo_sizes = []
        
        for repo_name in self.list_all_repositories():
            repo_data = self.get_repository_data(repo_name)
            if repo_data:
                size_bytes = repo_data.get('file_structure', {}).get('total_size', 0)
                repo_sizes.append({
                    'repository': repo_name,
                    'size_mb': size_bytes / (1024 * 1024),
                    'file_count': repo_data.get('file_structure', {}).get('total_files', 0),
                    'languages': repo_data.get('file_structure', {}).get('languages', {}),
                    'technologies': repo_data.get('technologies', {})
                })
        
        repo_sizes.sort(key=lambda x: x['size_mb'], reverse=True)
        return repo_sizes[:limit]


# MCP Tool Definitions for the enhanced functionality
def get_enhanced_mcp_tools() -> List[types.Tool]:
    """Get the list of enhanced MCP tools that use preprocessed data."""
    return [
        types.Tool(
            name="search_repositories_by_language",
            description="Find repositories that use a specific programming language, ranked by usage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Programming language to search for (e.g., Python, JavaScript)"}
                },
                "required": ["language"]
            }
        ),
        types.Tool(
            name="search_repositories_by_framework",
            description="Find repositories that use a specific framework or technology.",
            inputSchema={
                "type": "object",
                "properties": {
                    "framework": {"type": "string", "description": "Framework to search for (e.g., React, Django, Flask)"}
                },
                "required": ["framework"]
            }
        ),
        types.Tool(
            name="search_readme_content",
            description="Search across all README files for specific content or keywords.",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Term to search for in README files"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 20}
                },
                "required": ["search_term"]
            }
        ),
        types.Tool(
            name="get_technology_statistics",
            description="Get comprehensive statistics about technology usage across all repositories.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="get_repository_recommendations",
            description="Get recommendations for repositories similar to a given repository based on technology stack.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_name": {"type": "string", "description": "Name of the repository to find similar repos for"},
                    "limit": {"type": "integer", "description": "Maximum number of recommendations", "default": 5}
                },
                "required": ["repository_name"]
            }
        ),
        types.Tool(
            name="get_largest_repositories",
            description="Get the largest repositories by file size with their technology details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of repositories to return", "default": 10}
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_repository_details",
            description="Get comprehensive details about a specific repository including all analyzed data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_name": {"type": "string", "description": "Name of the repository to get details for"}
                },
                "required": ["repository_name"]
            }
        )
    ]


# Tool handler implementations
async def handle_enhanced_tool_call(tools: EnhancedRepositoryTools, name: str, arguments: dict) -> List[types.TextContent]:
    """Handle enhanced tool calls using preprocessed data."""
    
    try:
        if name == "search_repositories_by_language":
            language = arguments["language"]
            results = tools.search_repositories_by_language(language)
            return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
        
        elif name == "search_repositories_by_framework":
            framework = arguments["framework"]
            results = tools.search_repositories_by_technology("frameworks", framework)
            return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
        
        elif name == "search_readme_content":
            search_term = arguments["search_term"]
            max_results = arguments.get("max_results", 20)
            results = tools.search_readme_content(search_term, max_results)
            return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
        
        elif name == "get_technology_statistics":
            stats = tools.get_technology_statistics()
            return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]
        
        elif name == "get_repository_recommendations":
            repo_name = arguments["repository_name"]
            limit = arguments.get("limit", 5)
            recommendations = tools.get_repository_recommendations(repo_name, limit)
            return [types.TextContent(type="text", text=json.dumps(recommendations, indent=2))]
        
        elif name == "get_largest_repositories":
            limit = arguments.get("limit", 10)
            largest = tools.get_largest_repositories(limit)
            return [types.TextContent(type="text", text=json.dumps(largest, indent=2))]
        
        elif name == "get_repository_details":
            repo_name = arguments["repository_name"]
            details = tools.get_repository_data(repo_name)
            if details:
                return [types.TextContent(type="text", text=json.dumps(details, indent=2))]
            else:
                return [types.TextContent(type="text", text=f"Repository '{repo_name}' not found.")]
        
        else:
            return [types.TextContent(type="text", text=f"Unknown enhanced tool: {name}")]
    
    except Exception as e:
        error_message = f"Error executing {name}: {str(e)}"
        return [types.TextContent(type="text", text=error_message)]


# Example integration function
def integrate_enhanced_tools_to_mcp_server():
    """
    Example of how to integrate enhanced tools into the main MCP server.
    
    This function shows how to modify the existing MCP server to use the
    enhanced tools alongside the original tools.
    """
    
    # This would be added to the main server file
    enhanced_tools = EnhancedRepositoryTools()
    
    # In the list_tools handler, you would add:
    enhanced_tool_list = get_enhanced_mcp_tools()
    
    # In the call_tool handler, you would add checks for enhanced tools:
    # if name in [tool.name for tool in enhanced_tool_list]:
    #     return await handle_enhanced_tool_call(enhanced_tools, name, arguments)
    
    return enhanced_tools, enhanced_tool_list


if __name__ == "__main__":
    # Demo of enhanced tools functionality
    tools = EnhancedRepositoryTools()
    
    print("Enhanced Repository Tools Demo")
    print("=" * 40)
    
    # Test technology statistics
    stats = tools.get_technology_statistics()
    print("Technology Statistics:")
    print(f"Total repositories: {stats.get('overview', {}).get('total_repositories', 0)}")
    print(f"Total files: {stats.get('overview', {}).get('total_files', 0):,}")
    
    # Test language search
    python_repos = tools.search_repositories_by_language("Python")
    print(f"\nPython repositories found: {len(python_repos)}")
    
    # Test README search
    ml_readmes = tools.search_readme_content("machine learning")
    print(f"README files mentioning 'machine learning': {len(ml_readmes)}")
    
    print("\nEnhanced tools are ready for integration!")