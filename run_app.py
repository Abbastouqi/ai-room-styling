#!/usr/bin/env python3
"""
Room Redesign Application Launcher
Starts both backend API and serves frontend
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def start_backend():
    """Start Flask backend API"""
    print("🚀 Starting backend API...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Start Flask app
    subprocess.run([
        sys.executable, "app.py"
    ])

def start_frontend():
    """Start frontend server (simple HTTP server)"""
    print("🌐 Starting frontend server...")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    # Start simple HTTP server
    subprocess.run([
        sys.executable, "-m", "http.server", "8080"
    ])

def open_browser():
    """Open browser after servers start"""
    time.sleep(3)  # Wait for servers to start
    print("🌍 Opening browser...")
    webbrowser.open("http://localhost:8080")

def main():
    """Main launcher"""
    print("=" * 60)
    print("🏠 AI Room Redesign Studio")
    print("=" * 60)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import torch
        import flask
        print("✅ Core dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install requirements:")
        print("  pip install -r backend/requirements.txt")
        sys.exit(1)
    
    print("Starting servers...")
    print("- Backend API: http://localhost:5000")
    print("- Frontend UI: http://localhost:8080")
    print()
    print("Press Ctrl+C to stop both servers")
    print()
    
    try:
        # Start backend in separate thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Start browser opener in separate thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Start frontend (blocking)
        start_frontend()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()