# Repository Data Preprocessing Setup Guide

This guide explains how to set up and use the comprehensive repository data preprocessing system for your MCP GitHub repositories tool.

## 🚀 Quick Start

1. **Run the preprocessing** (this analyzes all your repositories):
   ```bash
   cd mcp-resources
   python run_preprocessing.py
   ```

2. **Query the results**:
   ```bash
   python query_data.py --report
   ```

3. **Use with MCP server**: The preprocessed data can now be used by your MCP server for fast queries.

## 📁 What Was Created

### Core Scripts
- **`data_preprocessor.py`** - Main preprocessing engine that analyzes repositories
- **`run_preprocessing.py`** - Easy-to-use runner script with various options
- **`query_data.py`** - Query utility for exploring the preprocessed data
- **`enhanced_mcp_tools.py`** - Enhanced MCP tools that use the preprocessed data

### Configuration & Documentation
- **`config.json`** - Configuration settings for preprocessing behavior
- **`requirements.txt`** - Python dependencies (minimal - uses mostly standard library)
- **`Makefile`** - Convenient commands for running preprocessing and managing data
- **`README.md`** - Detailed technical documentation
- **`SETUP_GUIDE.md`** - This setup guide

### Generated Data Files (after running preprocessing)
- **`repositories_data.json`** - Complete dataset with all analyzed information
- **`repositories_summary.json`** - High-level summary with aggregated statistics
- **`readme_index.json`** - Searchable index of all README files
- **`repositories/`** - Individual JSON files for each repository

## 🔧 Configuration

Edit `config.json` to customize:

```json
{
  "preprocessing": {
    "root_directory": "/home/snow/Documents/Projects/github-repositories/bkocis",
    "output_directory": "mcp-resources",
    "skip_directories": [".git", "node_modules", "__pycache__", ...],
    "max_file_size_mb": 10
  },
  "analysis": {
    "enable_git_analysis": true,
    "enable_dependency_analysis": true,
    "summary_max_chars": 500
  }
}
```

## 📊 What Gets Analyzed

For each repository, the system extracts:

### 📄 README Files
- Full content and summaries
- Word counts and metadata
- Searchable index across all repositories

### 🏗️ Code Structure
- File types and programming languages
- Directory organization
- File size statistics and largest files

### 🛠️ Technology Stack
- **Frameworks**: React, Django, Flask, Express.js, etc.
- **Tools**: Webpack, Babel, ESLint, Prettier, etc.
- **Databases**: PostgreSQL, MongoDB, Redis, etc.
- **Testing**: Jest, Pytest, Cypress, etc.
- **Deployment**: Docker, GitHub Actions, etc.

### 📈 Git Information
- Current branch and remote URL
- Latest commit details (author, date, message)
- Repository activity indicators

## 🎯 Usage Examples

### Basic Preprocessing
```bash
# Full analysis
make preprocess

# Quick analysis (faster, less detailed)
make quick

# See what would be processed
make dry-run
```

### Querying Data
```bash
# Generate comprehensive report
python query_data.py --report

# Find Python repositories
python query_data.py --language Python

# Find Flask applications
python query_data.py --framework Flask

# Search README files
python query_data.py --search-readme "machine learning"

# Show technology statistics
python query_data.py --tech-stats
```

### Using Makefile Commands
```bash
make help           # Show all available commands
make status         # Show current preprocessing status
make view-summary   # Display summary statistics
make view-repos     # List all processed repositories
make clean          # Clean output directory
make validate       # Validate output files
```

## 🔍 Query Capabilities

The preprocessed data enables powerful queries:

1. **Language-based filtering**: Find all repositories using specific programming languages
2. **Technology stack analysis**: Identify repositories using particular frameworks or tools
3. **Content search**: Search across all README files for keywords or concepts
4. **Size and complexity metrics**: Find largest repositories, most complex codebases
5. **Activity tracking**: Identify most recently updated repositories
6. **Similarity matching**: Find repositories with similar technology stacks

## 🚀 Integration with MCP Server

### Option 1: Direct Integration
Add the enhanced tools to your existing MCP server:

```python
from enhanced_mcp_tools import EnhancedRepositoryTools, get_enhanced_mcp_tools

# In your server setup
enhanced_tools = EnhancedRepositoryTools()
enhanced_tool_list = get_enhanced_mcp_tools()

# Add to your tool handlers
```

### Option 2: Hybrid Approach
Use preprocessed data for fast queries, fall back to live analysis for detailed operations:

```python
# Fast queries use preprocessed data
results = enhanced_tools.search_repositories_by_language("Python")

# Detailed analysis uses live file reading
content = get_readme_content(specific_path)
```

## 📈 Performance Benefits

### Before Preprocessing
- Scanning 84 repositories takes ~30-60 seconds
- Each query requires filesystem traversal
- Limited by I/O operations and git command execution

### After Preprocessing
- Initial preprocessing: ~2-3 minutes (one-time cost)
- Subsequent queries: ~50-100ms
- Rich filtering and search capabilities
- Aggregated statistics instantly available

## 🔄 Updating Data

### Incremental Updates
```bash
# Re-run preprocessing to update data
python run_preprocessing.py

# The system detects changes and updates affected repositories
```

### Automated Updates
Set up a cron job or schedule to run preprocessing periodically:
```bash
# Add to crontab for daily updates at 2 AM
0 2 * * * cd /path/to/mcp-resources && python run_preprocessing.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **Permission denied**: Ensure read access to all repository directories
2. **Git timeout**: Increase `max_git_timeout_seconds` in config.json
3. **Large repositories**: Consider excluding very large files or directories
4. **Memory usage**: For many repositories, consider processing in batches

### Debug Mode
```bash
# Run with verbose output
python run_preprocessing.py --verbose

# Test individual components
python -c "import data_preprocessor; print('✓ Import successful')"
```

## 📊 Sample Output

Your preprocessing results show:
- **84 repositories** analyzed
- **4,709 files** processed
- **2.1 GB** total codebase size
- **Top languages**: Python (773 files), HTML (117), Shell (66)
- **Frameworks found**: Flask, FastAPI, Express.js

## 🎁 Additional Features

### Technology Trend Analysis
```python
# Analyze technology adoption over time
stats = tools.get_technology_statistics()
print(f"Python adoption: {stats['languages']['Python']['percentage']}%")
```

### Repository Recommendations
```python
# Find similar repositories
recommendations = tools.get_repository_recommendations("my-flask-app")
```

### Custom Queries
```python
# Build custom analysis
from query_data import RepositoryDataQuery
query = RepositoryDataQuery()

# Find all machine learning projects
ml_repos = query.search_readmes("machine learning")
```

## 🔮 Future Enhancements

Potential improvements you could add:

1. **Semantic search**: Use embeddings for better README search
2. **Code quality metrics**: Integrate linting and complexity analysis
3. **Dependency graphs**: Visualize relationships between repositories
4. **Change tracking**: Monitor repository evolution over time
5. **Integration APIs**: Connect with GitHub API for additional metadata

---

## 🎉 You're All Set!

Your repository preprocessing system is now ready. The data has been analyzed and you can:

1. ✅ Query repositories by technology stack
2. ✅ Search across all README files
3. ✅ Get instant statistics and summaries
4. ✅ Integrate with your MCP server for enhanced performance
5. ✅ Build custom analysis tools using the structured data

Run `make status` to see your current data status, or `python query_data.py --report` for a comprehensive overview of your repositories!