#!/usr/bin/env python3
"""
Clean build artifacts and temporary files
Usage: python clean_build.py
"""

import shutil
import os
from pathlib import Path

def clean_build():
    """Remove build artifacts"""
    base_dir = Path(__file__).parent.absolute()
    
    dirs_to_remove = [
        'build',
        'dist',
        '__pycache__',
        'unified_app_v2.build',
        'unified_app_v2.dist',
        'unified_app_v2.onefile-build',
    ]
    
    files_to_remove = [
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.nuitka',
    ]
    
    print("="*80)
    print("Cleaning Build Artifacts")
    print("="*80 + "\n")
    
    removed_count = 0
    
    # Remove directories
    for dir_name in dirs_to_remove:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Removed directory: {dir_name}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {dir_name}: {e}")
    
    # Remove files matching patterns
    for pattern in files_to_remove:
        for file_path in base_dir.rglob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"✓ Removed file: {file_path.relative_to(base_dir)}")
                    removed_count += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"✓ Removed directory: {file_path.relative_to(base_dir)}")
                    removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {file_path}: {e}")
    
    print("\n" + "="*80)
    if removed_count > 0:
        print(f"Cleaned {removed_count} items")
    else:
        print("Nothing to clean - build directory is already clean")
    print("="*80 + "\n")

if __name__ == '__main__':
    clean_build()
