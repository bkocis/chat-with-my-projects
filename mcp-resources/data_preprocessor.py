#!/usr/bin/env python3
"""
Repository Data Preprocessing Script

This script analyzes repositories in a given directory and creates structured data
for the MCP GitHub repositories tool. It performs comprehensive analysis including:

- README extraction and processing
- Code structure analysis
- Language and framework detection
- Dependency analysis
- Repository metadata extraction
- File type distribution analysis

The output is stored in JSON format for efficient access by the MCP server.
"""

import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter
import hashlib

class RepositoryAnalyzer:
    """Analyzes a single repository and extracts comprehensive metadata."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo_name = self.repo_path.name
        
    def is_git_repo(self) -> bool:
        """Check if the directory is a git repository."""
        return (self.repo_path / '.git').exists()
    
    def get_readme_files(self) -> List[Dict[str, Any]]:
        """Find and analyze all README files in the repository."""
        readme_files = []
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and file_path.name.lower().startswith('readme'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract first paragraph as summary
                    lines = content.strip().split('\n')
                    summary_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            summary_lines.append(line)
                            if len(summary_lines) >= 3:
                                break
                    
                    summary = ' '.join(summary_lines)[:500] + ('...' if len(' '.join(summary_lines)) > 500 else '')
                    
                    readme_files.append({
                        'path': str(file_path.relative_to(self.repo_path)),
                        'absolute_path': str(file_path),
                        'size': file_path.stat().st_size,
                        'content': content,
                        'summary': summary,
                        'line_count': len(lines),
                        'word_count': len(content.split()),
                        'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        return readme_files
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze the file structure and types in the repository."""
        file_stats = {
            'total_files': 0,
            'total_size': 0,
            'file_types': Counter(),
            'languages': Counter(),
            'directories': [],
            'largest_files': []
        }
        
        # Language mappings
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'SASS',
            '.less': 'LESS',
            '.sql': 'SQL',
            '.sh': 'Shell',
            '.bash': 'Bash',
            '.zsh': 'Zsh',
            '.ps1': 'PowerShell',
            '.r': 'R',
            '.m': 'MATLAB',
            '.dart': 'Dart',
            '.vue': 'Vue',
            '.jsx': 'React',
            '.tsx': 'React TypeScript'
        }
        
        files_by_size = []
        
        for file_path in self.repo_path.rglob('*'):
            # Skip hidden directories and common build/cache directories
            if any(part.startswith('.') or part in ['node_modules', '__pycache__', 'build', 'dist', 'target'] 
                   for part in file_path.parts):
                continue
                
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    file_stats['total_files'] += 1
                    file_stats['total_size'] += size
                    
                    # File extension analysis
                    ext = file_path.suffix.lower()
                    file_stats['file_types'][ext] += 1
                    
                    # Language detection
                    if ext in language_map:
                        file_stats['languages'][language_map[ext]] += 1
                    
                    # Track large files
                    files_by_size.append((str(file_path.relative_to(self.repo_path)), size))
                    
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
            
            elif file_path.is_dir() and file_path != self.repo_path:
                file_stats['directories'].append(str(file_path.relative_to(self.repo_path)))
        
        # Get top 10 largest files
        files_by_size.sort(key=lambda x: x[1], reverse=True)
        file_stats['largest_files'] = files_by_size[:10]
        
        return file_stats
    
    def detect_frameworks_and_tools(self) -> Dict[str, List[str]]:
        """Detect frameworks, tools, and technologies used in the repository."""
        detected = {
            'frameworks': [],
            'tools': [],
            'databases': [],
            'deployment': [],
            'testing': [],
            'build_systems': []
        }
        
        # Check common config files and their contents
        config_files = {
            'package.json': self._analyze_package_json,
            'requirements.txt': self._analyze_requirements_txt,
            'Pipfile': self._analyze_pipfile,
            'pom.xml': self._analyze_pom_xml,
            'build.gradle': self._analyze_gradle,
            'Cargo.toml': self._analyze_cargo_toml,
            'composer.json': self._analyze_composer_json,
            'Gemfile': self._analyze_gemfile,
            'Dockerfile': self._analyze_dockerfile,
            'docker-compose.yml': self._analyze_docker_compose,
            '.github/workflows': self._analyze_github_actions,
            'Makefile': self._analyze_makefile
        }
        
        for config_file, analyzer in config_files.items():
            file_path = self.repo_path / config_file
            if file_path.exists():
                try:
                    result = analyzer(file_path)
                    for category, items in result.items():
                        if category in detected:
                            detected[category].extend(items)
                except Exception as e:
                    print(f"Error analyzing {config_file}: {e}")
        
        # Remove duplicates
        for category in detected:
            detected[category] = list(set(detected[category]))
        
        return detected
    
    def _analyze_package_json(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze package.json for Node.js dependencies."""
        result = defaultdict(list)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Framework detection
            deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
            
            framework_patterns = {
                'react': 'React',
                'vue': 'Vue.js',
                'angular': 'Angular',
                'express': 'Express.js',
                'next': 'Next.js',
                'nuxt': 'Nuxt.js',
                'gatsby': 'Gatsby',
                'svelte': 'Svelte',
                'fastify': 'Fastify'
            }
            
            tool_patterns = {
                'webpack': 'Webpack',
                'vite': 'Vite',
                'rollup': 'Rollup',
                'babel': 'Babel',
                'eslint': 'ESLint',
                'prettier': 'Prettier',
                'typescript': 'TypeScript'
            }
            
            test_patterns = {
                'jest': 'Jest',
                'mocha': 'Mocha',
                'chai': 'Chai',
                'cypress': 'Cypress',
                'playwright': 'Playwright'
            }
            
            for dep in deps:
                for pattern, name in framework_patterns.items():
                    if pattern in dep.lower():
                        result['frameworks'].append(name)
                
                for pattern, name in tool_patterns.items():
                    if pattern in dep.lower():
                        result['tools'].append(name)
                
                for pattern, name in test_patterns.items():
                    if pattern in dep.lower():
                        result['testing'].append(name)
        
        except Exception as e:
            print(f"Error parsing package.json: {e}")
        
        return result
    
    def _analyze_requirements_txt(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze requirements.txt for Python dependencies."""
        result = defaultdict(list)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            framework_patterns = {
                'django': 'Django',
                'flask': 'Flask',
                'fastapi': 'FastAPI',
                'tornado': 'Tornado',
                'pyramid': 'Pyramid',
                'streamlit': 'Streamlit',
                'gradio': 'Gradio'
            }
            
            tool_patterns = {
                'pandas': 'Pandas',
                'numpy': 'NumPy',
                'scipy': 'SciPy',
                'scikit-learn': 'Scikit-learn',
                'tensorflow': 'TensorFlow',
                'pytorch': 'PyTorch',
                'keras': 'Keras'
            }
            
            for pattern, name in framework_patterns.items():
                if pattern in content:
                    result['frameworks'].append(name)
            
            for pattern, name in tool_patterns.items():
                if pattern in content:
                    result['tools'].append(name)
        
        except Exception as e:
            print(f"Error parsing requirements.txt: {e}")
        
        return result
    
    def _analyze_dockerfile(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze Dockerfile for deployment information."""
        result = defaultdict(list)
        result['deployment'].append('Docker')
        return result
    
    def _analyze_docker_compose(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze docker-compose.yml for deployment and database information."""
        result = defaultdict(list)
        result['deployment'].append('Docker Compose')
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().lower()
            
            db_patterns = {
                'postgres': 'PostgreSQL',
                'mysql': 'MySQL',
                'mongo': 'MongoDB',
                'redis': 'Redis',
                'elasticsearch': 'Elasticsearch'
            }
            
            for pattern, name in db_patterns.items():
                if pattern in content:
                    result['databases'].append(name)
        
        except Exception as e:
            print(f"Error parsing docker-compose.yml: {e}")
        
        return result
    
    def _analyze_makefile(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze Makefile for build system information."""
        result = defaultdict(list)
        result['build_systems'].append('Make')
        return result
    
    # Stub methods for other analyzers
    def _analyze_pipfile(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def _analyze_pom_xml(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def _analyze_gradle(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def _analyze_cargo_toml(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def _analyze_composer_json(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def _analyze_gemfile(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def _analyze_github_actions(self, file_path: Path) -> Dict[str, List[str]]:
        return defaultdict(list)
    
    def get_git_info(self) -> Dict[str, Any]:
        """Extract git repository information."""
        git_info = {}
        
        if not self.is_git_repo():
            return git_info
        
        try:
            # Get current branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info['current_branch'] = result.stdout.strip()
            
            # Get last commit info
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%H|%an|%ae|%ad|%s'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split('|')
                if len(parts) == 5:
                    git_info['last_commit'] = {
                        'hash': parts[0],
                        'author_name': parts[1],
                        'author_email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    }
            
            # Get remote URL
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
        
        except Exception as e:
            print(f"Error getting git info for {self.repo_path}: {e}")
        
        return git_info
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the repository."""
        print(f"Analyzing repository: {self.repo_name}")
        
        analysis = {
            'repository_name': self.repo_name,
            'repository_path': str(self.repo_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'is_git_repo': self.is_git_repo(),
            'readme_files': self.get_readme_files(),
            'file_structure': self.analyze_file_structure(),
            'technologies': self.detect_frameworks_and_tools(),
            'git_info': self.get_git_info()
        }
        
        # Generate a content hash for change detection
        content_str = json.dumps(analysis, sort_keys=True)
        analysis['content_hash'] = hashlib.md5(content_str.encode()).hexdigest()
        
        return analysis


class DataPreprocessor:
    """Main data preprocessing coordinator."""
    
    def __init__(self, root_dir: str, output_dir: str = None):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir) if output_dir else Path('mcp-resources')
        self.output_dir.mkdir(exist_ok=True)
        
    def find_repositories(self) -> List[Path]:
        """Find all repositories in the root directory."""
        repositories = []
        
        for item in self.root_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                repositories.append(item)
        
        print(f"Found {len(repositories)} potential repositories")
        return repositories
    
    def process_repositories(self) -> Dict[str, Any]:
        """Process all repositories and create structured data."""
        repositories = self.find_repositories()
        processed_data = {
            'metadata': {
                'total_repositories': len(repositories),
                'processing_timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'version': '1.0.0'
            },
            'repositories': {},
            'summary': {
                'languages': Counter(),
                'frameworks': Counter(),
                'tools': Counter(),
                'total_files': 0,
                'total_size': 0
            }
        }
        
        for repo_path in repositories:
            try:
                analyzer = RepositoryAnalyzer(repo_path)
                analysis = analyzer.analyze()
                
                repo_name = repo_path.name
                processed_data['repositories'][repo_name] = analysis
                
                # Update summary statistics
                file_structure = analysis.get('file_structure', {})
                processed_data['summary']['total_files'] += file_structure.get('total_files', 0)
                processed_data['summary']['total_size'] += file_structure.get('total_size', 0)
                
                # Aggregate language stats
                for lang, count in file_structure.get('languages', {}).items():
                    processed_data['summary']['languages'][lang] += count
                
                # Aggregate technology stats
                technologies = analysis.get('technologies', {})
                for framework in technologies.get('frameworks', []):
                    processed_data['summary']['frameworks'][framework] += 1
                for tool in technologies.get('tools', []):
                    processed_data['summary']['tools'][tool] += 1
                
            except Exception as e:
                print(f"Error processing repository {repo_path}: {e}")
                processed_data['repositories'][repo_path.name] = {
                    'error': str(e),
                    'repository_name': repo_path.name,
                    'repository_path': str(repo_path)
                }
        
        return processed_data
    
    def save_processed_data(self, data: Dict[str, Any]) -> None:
        """Save processed data to JSON files."""
        # Save complete data
        output_file = self.output_dir / 'repositories_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved complete data to {output_file}")
        
        # Save summary data separately for quick access
        summary_file = self.output_dir / 'repositories_summary.json'
        summary_data = {
            'metadata': data['metadata'],
            'summary': data['summary'],
            'repository_list': list(data['repositories'].keys())
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved summary data to {summary_file}")
        
        # Save individual repository files for efficient access
        repos_dir = self.output_dir / 'repositories'
        repos_dir.mkdir(exist_ok=True)
        
        for repo_name, repo_data in data['repositories'].items():
            repo_file = repos_dir / f"{repo_name}.json"
            with open(repo_file, 'w', encoding='utf-8') as f:
                json.dump(repo_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved individual repository files to {repos_dir}")
    
    def generate_readme_index(self, data: Dict[str, Any]) -> None:
        """Generate a searchable index of all README files."""
        readme_index = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_readmes': 0
            },
            'readmes': []
        }
        
        for repo_name, repo_data in data['repositories'].items():
            if 'readme_files' in repo_data:
                for readme in repo_data['readme_files']:
                    readme_entry = {
                        'repository': repo_name,
                        'file_path': readme['path'],
                        'absolute_path': readme['absolute_path'],
                        'summary': readme['summary'],
                        'word_count': readme['word_count'],
                        'last_modified': readme['last_modified']
                    }
                    readme_index['readmes'].append(readme_entry)
                    readme_index['metadata']['total_readmes'] += 1
        
        index_file = self.output_dir / 'readme_index.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(readme_index, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Generated README index: {index_file}")
    
    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        print(f"Starting data preprocessing for: {self.root_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Process all repositories
        processed_data = self.process_repositories()
        
        # Save processed data
        self.save_processed_data(processed_data)
        
        # Generate README index
        self.generate_readme_index(processed_data)
        
        # Print summary
        metadata = processed_data['metadata']
        summary = processed_data['summary']
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Total repositories processed: {metadata['total_repositories']}")
        print(f"Total files analyzed: {summary['total_files']:,}")
        print(f"Total size: {summary['total_size'] / (1024*1024):.1f} MB")
        print(f"Top languages: {dict(summary['languages'].most_common(5))}")
        print(f"Top frameworks: {dict(summary['frameworks'].most_common(5))}")
        print("="*50)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess GitHub repositories for MCP tool')
    parser.add_argument(
        '--root-dir',
        default='/home/snow/Documents/Projects/github-repositories/bkocis',
        help='Root directory containing repositories'
    )
    parser.add_argument(
        '--output-dir',
        default='mcp-resources',
        help='Output directory for processed data'
    )
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.root_dir, args.output_dir)
    preprocessor.run()


if __name__ == '__main__':
    main()