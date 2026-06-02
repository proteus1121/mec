#!/usr/bin/env python3
"""
Build script for creating Windows executable from unified_app_v2.py using Nuitka
Requirements:
- Python 3.7+
- Nuitka
- All dependencies from requirements.txt

Usage:
    python build_windows.py

This will create a standalone Windows executable in the 'dist' folder.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_nuitka():
    """Check if Nuitka is installed"""
    try:
        result = subprocess.run(['python', '-m', 'nuitka', '--version'], 
                              capture_output=True, text=True)
        print(f"✓ Nuitka found: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"✗ Nuitka not found: {e}")
        return False

def install_nuitka():
    """Install Nuitka if not present"""
    print("Installing Nuitka...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'nuitka'], check=True)
        print("✓ Nuitka installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install Nuitka: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\nChecking dependencies...")
    try:
        import numpy
        print(f"✓ numpy: {numpy.__version__}")
    except ImportError:
        print("✗ numpy not found")
        return False
    
    try:
        import scipy
        print(f"✓ scipy: {scipy.__version__}")
    except ImportError:
        print("✗ scipy not found")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib not found")
        return False
    
    try:
        import statsmodels
        print(f"✓ statsmodels: {statsmodels.__version__}")
    except ImportError:
        print("✗ statsmodels not found")
        return False
    
    try:
        import filterpy
        print(f"✓ filterpy: {filterpy.__version__}")
    except ImportError:
        print("✗ filterpy not found")
        return False
    
    return True

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def build_with_nuitka():
    """Build the application using Nuitka"""
    print("\n" + "="*80)
    print("Starting Nuitka build process...")
    print("="*80 + "\n")
    
    # Get the current directory
    base_dir = Path(__file__).parent.absolute()
    source_file = base_dir / "unified_app_v2.py"
    
    if not source_file.exists():
        print(f"✗ Source file not found: {source_file}")
        return False
    
    # Nuitka command with optimized settings for Windows
    nuitka_cmd = [
        sys.executable,
        '-m', 'nuitka',
        
        # Basic options
        '--standalone',  # Create standalone distribution
        '--onefile',     # Create single executable file
        
        # Windows specific
        '--windows-disable-console',  # No console window (GUI app)
        '--enable-plugin=tk-inter',   # Enable tkinter support
        
        # Performance optimizations
        '--lto=yes',  # Link Time Optimization
        
        # Output settings
        '--output-dir=build',
        '--output-filename=UnifiedApp.exe',
        
        # Package inclusions
        '--include-package=numpy',
        '--include-package=scipy',
        '--include-package=matplotlib',
        '--include-package=statsmodels',
        '--include-package=filterpy',
        '--include-package=tkinter',
        '--include-module=matplotlib.backends.backend_tkagg',
        
        # Data files (if needed)
        '--include-data-files=cable_params.json=cable_params.json',
        
        # Follow imports
        '--follow-imports',
        
        # Progress
        '--show-progress',
        '--show-memory',
        
        # Target file
        str(source_file)
    ]
    
    # Check if cable_params.json exists, if not, remove that flag
    if not (base_dir / "cable_params.json").exists():
        nuitka_cmd = [cmd for cmd in nuitka_cmd if not cmd.startswith('--include-data-files')]
        print("ℹ cable_params.json not found, building without it")
    
    print("Build command:")
    print(" ".join(nuitka_cmd))
    print("\nThis may take several minutes...\n")
    
    try:
        # Run Nuitka build
        result = subprocess.run(nuitka_cmd, cwd=str(base_dir))
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("✓ Build completed successfully!")
            print("="*80)
            
            # Check for output file
            output_file = base_dir / "build" / "UnifiedApp.exe"
            if output_file.exists():
                print(f"\n✓ Executable created: {output_file}")
                print(f"  File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
                
                # Create a dist folder and copy the exe there
                dist_folder = base_dir / "dist"
                dist_folder.mkdir(exist_ok=True)
                
                final_exe = dist_folder / "UnifiedApp.exe"
                shutil.copy2(output_file, final_exe)
                print(f"\n✓ Executable copied to: {final_exe}")
                
                return True
            else:
                print(f"\n⚠ Build completed but executable not found at expected location")
                print(f"  Expected: {output_file}")
                return False
        else:
            print("\n" + "="*80)
            print(f"✗ Build failed with return code: {result.returncode}")
            print("="*80)
            return False
            
    except Exception as e:
        print(f"\n✗ Build error: {e}")
        return False

def main():
    """Main build function"""
    print("="*80)
    print("Windows Executable Build Script - Using Nuitka")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print("="*80 + "\n")
    
    # Check if Nuitka is installed
    if not check_nuitka():
        print("\nNuitka is not installed. Installing...")
        if not install_nuitka():
            print("\n✗ Failed to install Nuitka. Please install manually:")
            print("  pip install nuitka")
            return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\nSome dependencies are missing. Installing...")
        if not install_dependencies():
            print("\n✗ Failed to install dependencies. Please install manually:")
            print("  pip install -r requirements.txt")
            return 1
        
        # Recheck after installation
        if not check_dependencies():
            print("\n✗ Dependencies still missing after installation.")
            return 1
    
    # Build the application
    if not build_with_nuitka():
        print("\n✗ Build failed!")
        return 1
    
    print("\n" + "="*80)
    print("Build process completed successfully!")
    print("="*80)
    print("\nYou can find your executable in the 'dist' folder:")
    print("  dist/UnifiedApp.exe")
    print("\nNote: This executable is for Windows. Transfer it to a Windows")
    print("      machine to run it.")
    print("="*80 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
