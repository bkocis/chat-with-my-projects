# Repository Data Preprocessing

This directory contains tools for preprocessing GitHub repositories to create structured data for the MCP (Model Context Protocol) server.

## Overview

The data preprocessing script analyzes repositories and extracts:

- **README files**: Content, summaries, and metadata
- **Code structure**: File types, languages, and directory organization
- **Framework detection**: Identifies frameworks, tools, and technologies used
- **Dependency analysis**: Extracts dependencies from various package managers
- **Git information**: Branch, commit history, and remote URLs
- **File statistics**: Sizes, counts, and largest files

## Files

- `data_preprocessor.py` - Main preprocessing script
- `config.json` - Configuration settings
- `requirements.txt` - Python dependencies (optional)
- `README.md` - This documentation

## Usage

### Basic Usage

```bash
# Run with default settings
python data_preprocessor.py

# Specify custom directories
python data_preprocessor.py --root-dir /path/to/repos --output-dir /path/to/output
```

### Command Line Options

- `--root-dir`: Root directory containing repositories (default: `/home/snow/Documents/Projects/github-repositories/bkocis`)
- `--output-dir`: Output directory for processed data (default: `mcp-resources`)

### Configuration

Edit `config.json` to customize:

- Directories and files to skip
- Language mappings
- Framework detection patterns
- Analysis settings
- File size limits

## Output Files

The script generates several output files:

### `repositories_data.json`
Complete dataset containing all analyzed information for every repository.

### `repositories_summary.json`
High-level summary with aggregated statistics and repository list.

### `readme_index.json`
Searchable index of all README files with summaries.

### `repositories/`
Directory containing individual JSON files for each repository.

## Data Structure

### Repository Analysis Schema

```json
{
  "repository_name": "string",
  "repository_path": "string",
  "analysis_timestamp": "ISO datetime",
  "is_git_repo": "boolean",
  "readme_files": [
    {
      "path": "relative/path/to/readme.md",
      "absolute_path": "absolute/path/to/readme.md",
      "size": "bytes",
      "content": "full content",
      "summary": "first few lines summary",
      "line_count": "number",
      "word_count": "number",
      "last_modified": "ISO datetime"
    }
  ],
  "file_structure": {
    "total_files": "number",
    "total_size": "bytes",
    "file_types": {"extension": "count"},
    "languages": {"language": "count"},
    "directories": ["list of directories"],
    "largest_files": [["path", "size"]]
  },
  "technologies": {
    "frameworks": ["list of frameworks"],
    "tools": ["list of tools"],
    "databases": ["list of databases"],
    "deployment": ["list of deployment tools"],
    "testing": ["list of testing frameworks"],
    "build_systems": ["list of build systems"]
  },
  "git_info": {
    "current_branch": "string",
    "last_commit": {
      "hash": "string",
      "author_name": "string",
      "author_email": "string",
      "date": "string",
      "message": "string"
    },
    "remote_url": "string"
  },
  "content_hash": "MD5 hash for change detection"
}
```

## Framework Detection

The script detects frameworks and tools by analyzing:

- `package.json` (Node.js/JavaScript)
- `requirements.txt` (Python)
- `Pipfile` (Python)
- `pom.xml` (Java/Maven)
- `build.gradle` (Java/Gradle)
- `Cargo.toml` (Rust)
- `composer.json` (PHP)
- `Gemfile` (Ruby)
- `Dockerfile` (Docker)
- `docker-compose.yml` (Docker Compose)
- `Makefile` (Make)
- `.github/workflows/` (GitHub Actions)

## Language Detection

Languages are detected based on file extensions using a comprehensive mapping in the configuration file. The analysis counts files by language and provides statistics.

## Performance Considerations

- The script processes repositories in sequence
- Large repositories may take longer to analyze
- Git operations have timeout protection (configurable)
- File size limits prevent processing very large files
- Hidden directories and common build folders are skipped

## Error Handling

- Continues processing even if individual repositories fail
- Logs errors for debugging
- Creates error entries in output for failed repositories
- Graceful handling of encoding issues and permission errors

## Integration with MCP Server

The processed data is designed to work with the MCP server (`github_repos_mcp_server_http_mcp_lib.py`). The MCP server can be enhanced to:

1. Load preprocessed data for faster queries
2. Search across repository summaries
3. Filter repositories by technology stack
4. Provide aggregated statistics
5. Enable semantic search across README content

## Extending the Script

To add new analysis features:

1. Add new analyzer methods to the `RepositoryAnalyzer` class
2. Update the configuration file with new patterns
3. Modify the output schema as needed
4. Update this documentation

## Examples

### Find all React repositories

```python
import json

with open('repositories_summary.json') as f:
    summary = json.load(f)

# Find repositories using React
react_repos = []
for repo_name in summary['repository_list']:
    with open(f'repositories/{repo_name}.json') as f:
        repo_data = json.load(f)
    
    if 'React' in repo_data.get('technologies', {}).get('frameworks', []):
        react_repos.append(repo_name)

print(f"Found {len(react_repos)} React repositories")
```

### Get repository statistics

```python
import json

with open('repositories_summary.json') as f:
    summary = json.load(f)

stats = summary['summary']
print(f"Total files: {stats['total_files']:,}")
print(f"Total size: {stats['total_size'] / (1024*1024):.1f} MB")
print(f"Top languages: {dict(stats['languages'])}")
```

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure the script has read access to all repository directories
2. **Git timeout**: Increase `max_git_timeout_seconds` in config if repositories are large
3. **Encoding errors**: The script handles most encoding issues gracefully
4. **Memory usage**: For very large repositories, consider increasing file size limits

### Debug Mode

Add debug prints or logging to track processing progress:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```