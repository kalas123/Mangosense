#!/usr/bin/env python3
"""
Deployment Script for Colab-Connected Mango Disease Detection System

This script helps set up the complete system:
1. Lightweight Flask frontend
2. Google Colab GPU backend
3. Configuration and testing

Usage:
    python deploy_colab_system.py --mode local
    python deploy_colab_system.py --mode cloud --platform heroku
    python deploy_colab_system.py --setup-colab
"""

import os
import sys
import json
import subprocess
import argparse
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional

class ColabSystemDeployer:
    """Deploy and manage the Colab-connected system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements = [
            'flask>=2.3.0',
            'flask-sqlalchemy>=3.0.0',
            'pillow>=9.0.0',
            'requests>=2.28.0',
            'python-dotenv>=0.19.0'
        ]
        self.colab_requirements = [
            'flask',
            'pyngrok',
            'rembg[gpu]',
            'grad-cam',
            'captum',
            'lime',
            'opencv-python-headless',
            'timm',
            'torch',
            'torchvision',
            'numpy',
            'pillow',
            'matplotlib',
            'seaborn'
        ]
    
    def create_requirements_txt(self):
        """Create requirements.txt for the Flask app"""
        req_file = self.project_root / 'requirements.txt'
        with open(req_file, 'w') as f:
            for req in self.requirements:
                f.write(f"{req}\n")
        print(f"✅ Created {req_file}")
    
    def create_env_template(self):
        """Create .env template file"""
        env_template = self.project_root / '.env.template'
        with open(env_template, 'w') as f:
            f.write("""# Mango Disease Detection App Configuration

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite:///mango_disease.db

# Colab Backend
COLAB_BACKEND_URL=https://your-ngrok-url.ngrok.io

# Optional: Logging
LOG_LEVEL=INFO
""")
        print(f"✅ Created {env_template}")
        print("📝 Copy this to .env and fill in your values")
    
    def setup_local_environment(self):
        """Set up local development environment"""
        print("🔧 Setting up local environment...")
        
        # Create virtual environment
        if not (self.project_root / 'venv').exists():
            subprocess.run([sys.executable, '-m', 'venv', 'venv'])
            print("✅ Created virtual environment")
        
        # Install requirements
        pip_cmd = str(self.project_root / 'venv' / 'Scripts' / 'pip') if os.name == 'nt' else str(self.project_root / 'venv' / 'bin' / 'pip')
        subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'])
        print("✅ Installed requirements")
        
        # Create directories
        os.makedirs(self.project_root / 'uploads', exist_ok=True)
        os.makedirs(self.project_root / 'models', exist_ok=True)
        print("✅ Created necessary directories")
    
    def create_colab_notebook(self):
        """Create the complete Colab notebook"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🥭 Mango Disease Detection - GPU Backend\n",
                        "\n",
                        "This notebook provides GPU-powered inference, explainability, and background removal.\n",
                        "\n",
                        "## 🚀 Quick Start\n",
                        "1. Enable GPU runtime (Runtime → Change runtime type → GPU)\n",
                        "2. Run all cells in order\n",
                        "3. Upload your trained models when prompted\n",
                        "4. Copy the ngrok URL for your Flask app\n",
                        "\n",
                        "---"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Install required packages\n",
                        "!pip install -q flask pyngrok rembg[gpu] grad-cam captum lime opencv-python-headless timm\n",
                        "\n",
                        "print(\"✅ Packages installed successfully!\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Check GPU availability\n",
                        "import torch\n",
                        "import sys\n",
                        "\n",
                        "print(f\"Python: {sys.version}\")\n",
                        "print(f\"PyTorch: {torch.__version__}\")\n",
                        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                        "\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                        "    print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
                        "else:\n",
                        "    print(\"⚠️ GPU not available - using CPU\")\n",
                        "\n",
                        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                        "print(f\"\\n🎯 Using device: {device}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Create the backend service\n",
                        "backend_code = '''\n",
                        "# Insert the complete colab_backend.py code here\n",
                        "# This would be the full content of colab_backend.py\n",
                        "'''\n",
                        "\n",
                        "with open('colab_backend.py', 'w') as f:\n",
                        "    # In practice, you would read the actual colab_backend.py file\n",
                        "    # or embed the complete code here\n",
                        "    f.write(backend_code)\n",
                        "\n",
                        "print(\"✅ Backend service created\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Upload trained models\n",
                        "from google.colab import files\n",
                        "import os\n",
                        "\n",
                        "os.makedirs('models', exist_ok=True)\n",
                        "\n",
                        "print(\"📁 Upload your trained models:\")\n",
                        "print(\"Supported formats: .pth, .pth.tar, .pt\")\n",
                        "print(\"Recommended naming: resnet50_model.pth, efficientnet_b0_model.pth\")\n",
                        "\n",
                        "uploaded = files.upload()\n",
                        "\n",
                        "for filename in uploaded.keys():\n",
                        "    if filename.endswith(('.pth', '.pt', '.pth.tar')):\n",
                        "        os.rename(filename, f'models/{filename}')\n",
                        "        print(f\"✅ {filename} uploaded to models/\")\n",
                        "\n",
                        "# List available models\n",
                        "models = [f for f in os.listdir('models') if f.endswith(('.pth', '.pt', '.pth.tar'))]\n",
                        "print(f\"\\n📊 Available models: {len(models)}\")\n",
                        "for model in models:\n",
                        "    size = os.path.getsize(f'models/{model}') / 1e6\n",
                        "    print(f\"  - {model} ({size:.1f} MB)\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Setup ngrok and start service\n",
                        "from pyngrok import ngrok\n",
                        "import subprocess\n",
                        "import threading\n",
                        "import time\n",
                        "import requests\n",
                        "\n",
                        "# Set your ngrok authtoken here\n",
                        "# Get it from: https://dashboard.ngrok.com/get-started/your-authtoken\n",
                        "NGROK_TOKEN = \"YOUR_NGROK_TOKEN_HERE\"  # Replace with your token\n",
                        "\n",
                        "if NGROK_TOKEN != \"YOUR_NGROK_TOKEN_HERE\":\n",
                        "    ngrok.set_auth_token(NGROK_TOKEN)\n",
                        "    print(\"✅ ngrok token set\")\n",
                        "else:\n",
                        "    print(\"⚠️ Please set your ngrok token above\")\n",
                        "\n",
                        "print(\"🚀 Starting backend service...\")\n",
                        "\n",
                        "# Start Flask service in background\n",
                        "def run_backend():\n",
                        "    subprocess.run([\"python\", \"colab_backend.py\"])\n",
                        "\n",
                        "backend_thread = threading.Thread(target=run_backend, daemon=True)\n",
                        "backend_thread.start()\n",
                        "\n",
                        "time.sleep(10)  # Wait for service to start\n",
                        "\n",
                        "# Test local service\n",
                        "try:\n",
                        "    response = requests.get('http://localhost:5000/health', timeout=10)\n",
                        "    if response.status_code == 200:\n",
                        "        print(\"✅ Backend service is running\")\n",
                        "        print(f\"Status: {response.json()}\")\n",
                        "    else:\n",
                        "        print(f\"⚠️ Service returned status: {response.status_code}\")\n",
                        "except Exception as e:\n",
                        "    print(f\"❌ Service not responding: {e}\")\n",
                        "\n",
                        "# Create public tunnel\n",
                        "try:\n",
                        "    public_url = ngrok.connect(5000)\n",
                        "    print(f\"\\n🌐 Public URL: {public_url}\")\n",
                        "    print(f\"\\n📋 API Endpoints:\")\n",
                        "    print(f\"  GET  {public_url}/health\")\n",
                        "    print(f\"  POST {public_url}/process_image\")\n",
                        "    print(f\"  GET  {public_url}/models\")\n",
                        "    \n",
                        "    print(f\"\\n💡 Configure your Flask app:\")\n",
                        "    print(f\"COLAB_BACKEND_URL = '{public_url}'\")\n",
                        "    \n",
                        "    # Test public endpoint\n",
                        "    test_response = requests.get(f\"{public_url}/health\", timeout=10)\n",
                        "    if test_response.status_code == 200:\n",
                        "        print(\"\\n✅ Public endpoint is accessible\")\n",
                        "    else:\n",
                        "        print(f\"\\n⚠️ Public endpoint returned: {test_response.status_code}\")\n",
                        "        \n",
                        "except Exception as e:\n",
                        "    print(f\"❌ ngrok tunnel failed: {e}\")\n",
                        "    print(\"Please check your authtoken and try again\")\n",
                        "\n",
                        "print(\"\\n🎉 Backend service is ready!\")\n",
                        "print(\"Keep this notebook running to maintain the service.\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Keep the service alive (run this cell to prevent timeout)\n",
                        "import time\n",
                        "from datetime import datetime\n",
                        "\n",
                        "print(\"🔄 Service keep-alive started\")\n",
                        "print(\"This will prevent the notebook from timing out\")\n",
                        "print(\"Press Ctrl+C to stop\")\n",
                        "\n",
                        "try:\n",
                        "    while True:\n",
                        "        current_time = datetime.now().strftime('%H:%M:%S')\n",
                        "        print(f\"⏰ {current_time} - Service running...\")\n",
                        "        \n",
                        "        # Test service health\n",
                        "        try:\n",
                        "            response = requests.get('http://localhost:5000/health', timeout=5)\n",
                        "            if response.status_code == 200:\n",
                        "                print(\"   ✅ Backend healthy\")\n",
                        "            else:\n",
                        "                print(f\"   ⚠️ Backend status: {response.status_code}\")\n",
                        "        except:\n",
                        "            print(\"   ❌ Backend not responding\")\n",
                        "        \n",
                        "        time.sleep(300)  # Check every 5 minutes\n",
                        "        \n",
                        "except KeyboardInterrupt:\n",
                        "    print(\"\\n🛑 Keep-alive stopped\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cleanup (run when finished)\n",
                        "def cleanup():\n",
                        "    print(\"🧹 Cleaning up resources...\")\n",
                        "    \n",
                        "    # Clear GPU memory\n",
                        "    if torch.cuda.is_available():\n",
                        "        torch.cuda.empty_cache()\n",
                        "        print(\"✅ GPU memory cleared\")\n",
                        "    \n",
                        "    # Close ngrok tunnels\n",
                        "    try:\n",
                        "        ngrok.kill()\n",
                        "        print(\"✅ ngrok tunnels closed\")\n",
                        "    except:\n",
                        "        pass\n",
                        "    \n",
                        "    print(\"✅ Cleanup completed\")\n",
                        "\n",
                        "# Uncomment to run cleanup\n",
                        "# cleanup()\n",
                        "\n",
                        "print(\"💡 Run cleanup() when you're finished with the service\")"
                    ]
                }
            ],
            "metadata": {
                "accelerator": "GPU",
                "colab": {
                    "gpuType": "T4",
                    "provenance": []
                },
                "kernelspec": {
                    "display_name": "Python 3",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
        notebook_file = self.project_root / 'Mango_Disease_Colab_Backend.ipynb'
        with open(notebook_file, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print(f"✅ Created {notebook_file}")
        print("📝 Upload this notebook to Google Colab")
    
    def create_heroku_config(self):
        """Create Heroku deployment files"""
        # Procfile
        procfile = self.project_root / 'Procfile'
        with open(procfile, 'w') as f:
            f.write("web: python lightweight_app.py\n")
        
        # runtime.txt
        runtime = self.project_root / 'runtime.txt'
        with open(runtime, 'w') as f:
            f.write("python-3.11.0\n")
        
        print("✅ Created Heroku configuration files")
        print("📝 Set COLAB_BACKEND_URL config var in Heroku dashboard")
    
    def create_railway_config(self):
        """Create Railway deployment files"""
        railway_toml = self.project_root / 'railway.toml'
        with open(railway_toml, 'w') as f:
            f.write("""[build]
builder = "nixpacks"

[deploy]
startCommand = "python lightweight_app.py"
healthcheckPath = "/"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 10

[env]
PORT = "5001"
""")
        
        print("✅ Created Railway configuration")
    
    def test_system(self, backend_url: str):
        """Test the complete system"""
        print(f"🧪 Testing system with backend: {backend_url}")
        
        # Test backend health
        try:
            response = requests.get(f"{backend_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Backend health check passed")
                health_data = response.json()
                print(f"   Device: {health_data.get('device')}")
                print(f"   CUDA: {health_data.get('cuda_available')}")
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend connection failed: {e}")
            return False
        
        # Test models endpoint
        try:
            response = requests.get(f"{backend_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                print(f"✅ Found {len(models)} available models")
                for model in models:
                    print(f"   - {model.get('name')} ({model.get('architecture')})")
            else:
                print(f"⚠️ Models endpoint returned: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Models endpoint failed: {e}")
        
        print("✅ System test completed")
        return True
    
    def deploy_local(self):
        """Deploy for local development"""
        print("🏠 Setting up local deployment...")
        
        self.create_requirements_txt()
        self.create_env_template()
        self.setup_local_environment()
        self.create_colab_notebook()
        
        print("\n🎉 Local setup complete!")
        print("\n📋 Next steps:")
        print("1. Copy .env.template to .env and configure")
        print("2. Upload Mango_Disease_Colab_Backend.ipynb to Google Colab")
        print("3. Run the Colab notebook and get ngrok URL")
        print("4. Set COLAB_BACKEND_URL in .env")
        print("5. Run: python lightweight_app.py")
    
    def deploy_cloud(self, platform: str):
        """Deploy to cloud platform"""
        print(f"☁️ Setting up {platform} deployment...")
        
        self.create_requirements_txt()
        self.create_env_template()
        self.create_colab_notebook()
        
        if platform == 'heroku':
            self.create_heroku_config()
            print("\n🎉 Heroku setup complete!")
            print("\n📋 Next steps:")
            print("1. heroku create your-app-name")
            print("2. heroku config:set COLAB_BACKEND_URL=your-ngrok-url")
            print("3. git push heroku main")
            
        elif platform == 'railway':
            self.create_railway_config()
            print("\n🎉 Railway setup complete!")
            print("\n📋 Next steps:")
            print("1. Connect your GitHub repo to Railway")
            print("2. Set COLAB_BACKEND_URL environment variable")
            print("3. Deploy!")
        
        else:
            print(f"❌ Unknown platform: {platform}")
    
    def setup_colab_instructions(self):
        """Display Colab setup instructions"""
        print("📚 Google Colab Setup Instructions")
        print("=" * 50)
        
        instructions = """
1. 🌐 Open Google Colab (colab.research.google.com)
2. 📁 Upload Mango_Disease_Colab_Backend.ipynb
3. ⚙️ Enable GPU runtime:
   - Runtime → Change runtime type
   - Hardware accelerator → GPU
   - Save
4. 🔑 Get ngrok token:
   - Go to ngrok.com and sign up
   - Get your authtoken from dashboard
5. ▶️ Run all notebook cells in order
6. 📋 Copy the ngrok URL when it appears
7. 🔧 Configure your Flask app with the URL
8. 🎉 Start analyzing images!

💡 Pro Tips:
- Use Colab Pro for longer runtimes
- Keep the notebook running to maintain service
- Run the keep-alive cell to prevent timeouts
- Upload models when prompted
        """
        
        print(instructions)

def main():
    parser = argparse.ArgumentParser(description='Deploy Colab-connected Mango Disease Detection System')
    parser.add_argument('--mode', choices=['local', 'cloud'], required=True,
                       help='Deployment mode')
    parser.add_argument('--platform', choices=['heroku', 'railway'],
                       help='Cloud platform (required for cloud mode)')
    parser.add_argument('--setup-colab', action='store_true',
                       help='Show Colab setup instructions')
    parser.add_argument('--test', type=str,
                       help='Test system with backend URL')
    
    args = parser.parse_args()
    
    deployer = ColabSystemDeployer()
    
    if args.setup_colab:
        deployer.setup_colab_instructions()
        return
    
    if args.test:
        deployer.test_system(args.test)
        return
    
    if args.mode == 'local':
        deployer.deploy_local()
    elif args.mode == 'cloud':
        if not args.platform:
            print("❌ Cloud mode requires --platform argument")
            sys.exit(1)
        deployer.deploy_cloud(args.platform)

if __name__ == '__main__':
    main()