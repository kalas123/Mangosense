#!/usr/bin/env python3
"""
Quick Start Script for Colab-Connected Mango Disease Detection

This script helps you get up and running quickly with the complete system.

Usage:
    python quick_start.py
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  🥭 MANGO LEAF DISEASE DETECTION - GPU POWERED SYSTEM 🥭                     ║
║                                                                               ║
║  ⚡ GPU-Accelerated Inference via Google Colab                               ║
║  🎨 AI Background Removal                                                     ║
║  🔍 Explainable AI (GradCAM, LIME, Integrated Gradients)                    ║
║  💊 Treatment Recommendations                                                 ║
║  🌐 Beautiful Web Interface                                                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_project_structure():
    """Create necessary directories"""
    directories = ['uploads', 'models', 'templates', 'static']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_requirements():
    """Install required packages"""
    requirements = [
        'flask>=2.3.0',
        'flask-sqlalchemy>=3.0.0',
        'pillow>=9.0.0',
        'requests>=2.28.0',
        'python-dotenv>=0.19.0'
    ]
    
    print("📦 Installing requirements...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            print(f"✅ Installed: {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {req}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    env_content = """# Mango Disease Detection App Configuration

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production

# Database
DATABASE_URL=sqlite:///mango_disease.db

# Colab Backend (Set this after setting up Colab)
COLAB_BACKEND_URL=

# Optional Settings
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("📝 You'll need to set COLAB_BACKEND_URL after setting up Colab")

def check_files_exist():
    """Check if required files exist"""
    required_files = [
        'lightweight_app.py',
        'colab_backend.py',
        'templates/lightweight_index.html',
        'templates/lightweight_results.html',
        'templates/configure_backend.html',
        'templates/status.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all files are present before running.")
        return False
    
    print("✅ All required files found")
    return True

def test_flask_app():
    """Test if Flask app can start"""
    try:
        # Import to check for syntax errors
        import importlib.util
        spec = importlib.util.spec_from_file_location("lightweight_app", "lightweight_app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✅ Flask app syntax check passed")
        return True
    except Exception as e:
        print(f"❌ Flask app has issues: {e}")
        return False

def display_colab_instructions():
    """Display Colab setup instructions"""
    instructions = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          GOOGLE COLAB SETUP                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🚀 STEP 1: Set up Google Colab Backend

1. 🌐 Go to: https://colab.research.google.com
2. 📁 Upload: Mango_Disease_Colab_Backend.ipynb (created for you)
3. ⚙️ Enable GPU:
   - Runtime → Change runtime type
   - Hardware accelerator → GPU
   - Save
4. 🔑 Get ngrok token:
   - Go to: https://ngrok.com
   - Sign up (free)
   - Get authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken
5. ▶️ Run all cells in the Colab notebook
6. 📋 Copy the ngrok URL (looks like: https://abc123.ngrok.io)

🔧 STEP 2: Configure Local App

1. 🖥️ Run: python lightweight_app.py
2. 🌐 Open: http://localhost:5001
3. ⚙️ Go to: Configure Backend
4. 📝 Paste your ngrok URL
5. ✅ Test connection

🎉 STEP 3: Start Analyzing!

Upload mango leaf images and get:
- 🎯 Disease predictions
- 🔍 AI explanations (heatmaps)
- 💊 Treatment recommendations
- ⚡ GPU-powered processing
    """
    
    print(instructions)

def start_flask_app():
    """Start the Flask application"""
    print("\n🚀 Starting Flask application...")
    print("📍 URL: http://localhost:5001")
    print("⏹️ Press Ctrl+C to stop")
    
    try:
        # Try to open browser
        webbrowser.open('http://localhost:5001')
        print("🌐 Opened browser automatically")
    except:
        print("🌐 Please open http://localhost:5001 in your browser")
    
    # Start Flask app
    try:
        subprocess.run([sys.executable, 'lightweight_app.py'])
    except KeyboardInterrupt:
        print("\n👋 Flask app stopped")

def main():
    """Main setup and start process"""
    print_banner()
    
    print("🔍 Checking system requirements...")
    check_python_version()
    
    print("\n📁 Setting up project structure...")
    create_project_structure()
    
    print("\n📦 Installing requirements...")
    install_requirements()
    
    print("\n⚙️ Creating configuration...")
    create_env_file()
    
    print("\n🔍 Checking files...")
    if not check_files_exist():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        return
    
    print("\n🧪 Testing Flask app...")
    if not test_flask_app():
        print("\n❌ Flask app has issues. Please check the code.")
        return
    
    print("\n✅ Local setup complete!")
    
    # Display Colab instructions
    display_colab_instructions()
    
    # Ask if user wants to start the app
    print("\n" + "="*80)
    response = input("🚀 Ready to start the Flask app? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        start_flask_app()
    else:
        print("\n📋 To start later, run: python lightweight_app.py")
        print("🌐 Then visit: http://localhost:5001")
        print("\n💡 Don't forget to set up your Colab backend first!")

if __name__ == '__main__':
    main()