#!/usr/bin/env python3
"""
Simple runner script for the data preprocessing pipeline.

This script provides an easy way to run the preprocessing with different
configurations and options.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the current directory to path to import data_preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import DataPreprocessor


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main runner function."""
    parser = argparse.ArgumentParser(
        description='Run repository data preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_preprocessing.py
  
  # Use custom config file
  python run_preprocessing.py --config my_config.json
  
  # Override root directory
  python run_preprocessing.py --root-dir /path/to/repos
  
  # Quick mode (minimal analysis)
  python run_preprocessing.py --quick
  
  # Verbose output
  python run_preprocessing.py --verbose
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    
    parser.add_argument(
        '--root-dir',
        help='Root directory containing repositories (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for processed data (overrides config)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip git analysis and detailed file scanning'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    preprocessing_config = config.get('preprocessing', {})
    
    # Determine root directory
    root_dir = (
        args.root_dir or 
        preprocessing_config.get('root_directory') or 
        '/home/snow/Documents/Projects/github-repositories/bkocis'
    )
    
    # Determine output directory
    output_dir = (
        args.output_dir or 
        preprocessing_config.get('output_directory') or 
        'mcp-resources'
    )
    
    # Validate root directory exists
    if not Path(root_dir).exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        sys.exit(1)
    
    print("Repository Data Preprocessing")
    print("=" * 40)
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {args.config}")
    print(f"Quick mode: {args.quick}")
    print(f"Verbose: {args.verbose}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 40)
    
    if args.dry_run:
        print("\nDRY RUN - Analyzing what would be processed:")
        
        # Just list repositories that would be processed
        root_path = Path(root_dir)
        repos = []
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                repos.append(item.name)
        
        print(f"\nFound {len(repos)} repositories:")
        for repo in sorted(repos):
            print(f"  - {repo}")
        
        print(f"\nWould create output in: {output_dir}")
        print("Run without --dry-run to process repositories.")
        return
    
    # Create and configure preprocessor
    preprocessor = DataPreprocessor(root_dir, output_dir)
    
    # Apply quick mode modifications if requested
    if args.quick:
        print("Quick mode enabled - some analysis will be skipped")
        # You could modify the preprocessor here to skip certain analyses
    
    try:
        # Run preprocessing
        preprocessor.run()
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Show quick summary of what was created
        output_path = Path(output_dir)
        files_created = []
        
        for file_path in output_path.glob('*.json'):
            files_created.append(file_path.name)
        
        if files_created:
            print(f"Files created: {', '.join(sorted(files_created))}")
        
        repos_dir = output_path / 'repositories'
        if repos_dir.exists():
            repo_files = list(repos_dir.glob('*.json'))
            print(f"Individual repository files: {len(repo_files)}")
    
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()