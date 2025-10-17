#!/usr/bin/env python3
# Simple test script for CAC components

import sys
import os
import json
import sqlite3
from pathlib import Path

def test_basic_app():
    """Test basic app functionality"""
    print("Testing basic app...")
    try:
        # Check if basic_app.py exists and can be imported
        sys.path.append('.')
        import basic_app
        print("✓ basic_app.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ basic_app.py failed: {e}")
        return False

def test_unified_api():
    """Test unified API functionality"""
    print("Testing unified API...")
    try:
        import unified_api
        print("✓ unified_api.py imports successfully")
        
        # Test database initialization
        db = unified_api.DatabaseManager("test_cac.db")
        print("✓ Database initialized")
        
        # Test user data creation
        user_data = unified_api.UserData(user_id="test_user")
        db.save_user_data(user_data)
        print("✓ User data saved")
        
        # Test user data retrieval
        retrieved = db.get_user_data("test_user")
        if retrieved:
            print("✓ User data retrieved")
        else:
            print("✗ User data retrieval failed")
            return False
            
        return True
    except Exception as e:
        print(f"✗ unified_api.py failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit app"""
    print("Testing Streamlit app...")
    try:
        import Streamlit
        print("✓ Streamlit.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ Streamlit.py failed: {e}")
        return False

def test_gait_tracker():
    """Test gait tracker component"""
    print("Testing gait tracker...")
    try:
        # Check if GaitTracker files exist
        if Path("GaitTracker.jsx").exists():
            print("✓ GaitTracker.jsx exists")
        else:
            print("✗ GaitTracker.jsx missing")
            return False
            
        if Path("GaitTracker_improved.jsx").exists():
            print("✓ GaitTracker_improved.jsx exists")
        else:
            print("✗ GaitTracker_improved.jsx missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Gait tracker test failed: {e}")
        return False

def test_config_files():
    """Test configuration files"""
    print("Testing config files...")
    try:
        if Path("requirements.txt").exists():
            print("✓ requirements.txt exists")
        else:
            print("✗ requirements.txt missing")
            return False
            
        if Path("package.json").exists():
            print("✓ package.json exists")
        else:
            print("✗ package.json missing")
            return False
            
        if Path("config.env").exists():
            print("✓ config.env exists")
        else:
            print("✗ config.env missing")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Config files test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing CAC Components")
    print("=" * 30)
    
    tests = [
        test_config_files,
        test_basic_app,
        test_unified_api,
        test_streamlit,
        test_gait_tracker
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 30)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
