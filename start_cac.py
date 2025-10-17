#!/usr/bin/env python3
# CAC Startup Script
# starts all components

import subprocess
import time
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    try:
        import flask
        print("✓ Flask installed")
    except ImportError:
        print("✗ Flask not installed - run: pip install flask")
        return False
    
    try:
        import streamlit
        print("✓ Streamlit installed")
    except ImportError:
        print("✗ Streamlit not installed - run: pip install streamlit")
        return False
    
    try:
        import websockets
        print("✓ WebSockets installed")
    except ImportError:
        print("✗ WebSockets not installed - run: pip install websockets")
        return False
    
    return True

def start_component(name, command, port=None):
    """Start a component in background"""
    print(f"Starting {name}...")
    try:
        if port:
            print(f"  → {name} will be available on port {port}")
        
        # Start process in background
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print(f"✓ {name} started successfully")
            return process
        else:
            print(f"✗ {name} failed to start")
            return None
            
    except Exception as e:
        print(f"✗ Error starting {name}: {e}")
        return None

def main():
    """Main startup function"""
    print("🚀 Starting CAC Walking Tracker")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first.")
        sys.exit(1)
    
    print("\nStarting components...")
    
    # Start WebSocket API server
    api_process = start_component(
        "WebSocket API", 
        "python3 unified_api.py",
        8765
    )
    
    # Start basic web app
    web_process = start_component(
        "Basic Web App", 
        "python3 basic_app.py",
        8000
    )
    
    # Start Streamlit dashboard
    streamlit_process = start_component(
        "Streamlit Dashboard", 
        "streamlit run Streamlit.py",
        8501
    )
    
    print("\n" + "=" * 40)
    print("🎯 All components started!")
    print("\nAvailable services:")
    print("  • Basic App: http://localhost:8000")
    print("  • Streamlit Dashboard: http://localhost:8501")
    print("  • WebSocket API: ws://localhost:8765")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if any process died
            if api_process and api_process.poll() is not None:
                print("⚠️  WebSocket API stopped unexpectedly")
                api_process = None
            
            if web_process and web_process.poll() is not None:
                print("⚠️  Basic Web App stopped unexpectedly")
                web_process = None
            
            if streamlit_process and streamlit_process.poll() is not None:
                print("⚠️  Streamlit Dashboard stopped unexpectedly")
                streamlit_process = None
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
        # Stop all processes
        for process in [api_process, web_process, streamlit_process]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
        
        print("✓ All services stopped")

if __name__ == "__main__":
    main()
