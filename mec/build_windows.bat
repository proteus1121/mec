@echo off
REM Build script for Windows - Creates executable from unified_app_v2.py
REM This script should be run on Windows with Python installed

echo ================================================================================
echo Windows Executable Build Script - Using Nuitka
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found
python --version
echo.

REM Run the Python build script
echo Running build script...
python build_windows.py

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Build completed successfully!
echo ================================================================================
echo.
echo Your executable is located at: dist\UnifiedApp.exe
echo.
pause
