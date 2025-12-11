#!/usr/bin/env python3
"""
F1 Prediction API - Development Runner
Quick script to start the development server
"""

import subprocess
import sys
import os

def main():
    # Change to the backend directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(backend_dir)
    
    # Check if virtual environment exists
    venv_python = os.path.join("f1env", "Scripts", "python.exe") if sys.platform == "win32" else os.path.join("f1env", "bin", "python")
    
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found!")
        print("   Run: python -m venv f1env")
        print("   Then: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🏎️  Starting F1 Prediction API...")
    print("=" * 50)
    
    # Run the API
    subprocess.run([venv_python, "-m", "app.api"])

if __name__ == "__main__":
    main()
