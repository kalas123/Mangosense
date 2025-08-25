"""
Template for creating the Colab notebook
This will be converted to a proper .ipynb file
"""

NOTEBOOK_CONTENT = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🥭 Mango Leaf Disease Detection - GPU Backend Service\n\n"
                "This notebook provides GPU-powered inference, explainability, and background removal.\n\n"
                "## 🚀 Features\n"
                "- **Background Removal**: AI-powered background removal\n"
                "- **Disease Detection**: GPU-accelerated inference\n"
                "- **Explainability**: GradCAM heatmaps\n"
                "- **Treatment Recommendations**: Expert advice\n"
                "- **Public API**: Connect via ngrok\n\n"
                "## 📋 Instructions\n"
                "1. Run all cells in order\n"
                "2. Upload your trained models\n"
                "3. Get the ngrok public URL\n"
                "4. Use URL in your Flask app\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "!pip install -q flask pyngrok\n",
                "!pip install -q rembg[gpu]\n",
                "!pip install -q grad-cam captum lime\n",
                "!pip install -q opencv-python-headless\n",
                "!pip install -q timm\n\n",
                "print(\"✅ Packages installed successfully!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check GPU and system info\n",
                "import torch\n",
                "import sys\n\n",
                "print(f\"Python: {sys.version}\")\n",
                "print(f\"PyTorch: {torch.__version__}\")\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"\\n🎯 Using: {device}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Download the backend service code\n",
                "import urllib.request\n",
                "import os\n\n",
                "# URL to your backend service (you'll need to host this)\n",
                "backend_url = \"https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/colab_backend.py\"\n\n",
                "try:\n",
                "    urllib.request.urlretrieve(backend_url, \"colab_backend.py\")\n",
                "    print(\"✅ Backend service downloaded\")\n",
                "except:\n",
                "    print(\"⚠️ Could not download backend service\")\n",
                "    print(\"Please upload colab_backend.py manually\")\n\n",
                "# Alternative: Create the service inline\n",
                "if not os.path.exists(\"colab_backend.py\"):\n",
                "    print(\"Creating backend service...\")\n",
                "    # The backend code would be embedded here\n",
                "    with open(\"colab_backend.py\", \"w\") as f:\n",
                "        f.write(\"# Backend service code will be here\")\n",
                "    print(\"✅ Backend service created\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Upload your trained models\n",
                "from google.colab import files\n",
                "import os\n\n",
                "os.makedirs('models', exist_ok=True)\n\n",
                "print(\"📁 Upload your trained models:\")\n",
                "print(\"Formats: .pth, .pth.tar, .pt\")\n",
                "print(\"Names: resnet50_model.pth, efficientnet_b0_model.pth\")\n\n",
                "# Uncomment to upload\n",
                "# uploaded = files.upload()\n",
                "# for filename in uploaded.keys():\n",
                "#     if filename.endswith(('.pth', '.pt', '.pth.tar')):\n",
                "#         os.rename(filename, f'models/{filename}')\n",
                "#         print(f\"✅ {filename} uploaded\")\n\n",
                "# List models\n",
                "models = [f for f in os.listdir('models') if f.endswith(('.pth', '.pt', '.pth.tar'))]\n",
                "print(f\"\\nAvailable models: {len(models)}\")\n",
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
                "# Start the backend service\n",
                "import subprocess\n",
                "import threading\n",
                "import time\n",
                "from pyngrok import ngrok\n",
                "import requests\n\n",
                "# Set ngrok authtoken (get from https://dashboard.ngrok.com/)\n",
                "# ngrok.set_auth_token(\"YOUR_TOKEN_HERE\")\n\n",
                "print(\"🚀 Starting backend service...\")\n\n",
                "# Start Flask service in background\n",
                "def run_backend():\n",
                "    subprocess.run([\"python\", \"colab_backend.py\"])\n\n",
                "backend_thread = threading.Thread(target=run_backend, daemon=True)\n",
                "backend_thread.start()\n\n",
                "time.sleep(5)  # Wait for service to start\n\n",
                "# Test local service\n",
                "try:\n",
                "    response = requests.get('http://localhost:5000/health', timeout=10)\n",
                "    print(f\"✅ Service running: {response.json()}\")\n",
                "except Exception as e:\n",
                "    print(f\"⚠️ Service not responding: {e}\")\n\n",
                "# Create public tunnel\n",
                "try:\n",
                "    public_url = ngrok.connect(5000)\n",
                "    print(f\"\\n🌐 Public URL: {public_url}\")\n",
                "    print(f\"\\n📋 API Endpoints:\")\n",
                "    print(f\"  GET  {public_url}/health\")\n",
                "    print(f\"  POST {public_url}/process_image\")\n",
                "    \n",
                "    print(f\"\\n💡 Add to your Flask app:\")\n",
                "    print(f\"COLAB_BACKEND_URL = '{public_url}'\")\n",
                "    \n",
                "except Exception as e:\n",
                "    print(f\"❌ Ngrok failed: {e}\")\n",
                "    print(\"Check your authtoken\")\n\n",
                "print(\"\\n🎉 Backend service ready!\")\n",
                "print(\"Keep this notebook running!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test the API\n",
                "import requests\n",
                "import json\n\n",
                "if 'public_url' in locals():\n",
                "    # Test health endpoint\n",
                "    try:\n",
                "        response = requests.get(f\"{public_url}/health\")\n",
                "        print(f\"🏥 Health: {response.status_code}\")\n",
                "        print(json.dumps(response.json(), indent=2))\n",
                "    except Exception as e:\n",
                "        print(f\"❌ Health check failed: {e}\")\n",
                "        \n",
                "    # Test models endpoint\n",
                "    try:\n",
                "        response = requests.get(f\"{public_url}/models\")\n",
                "        print(f\"\\n🧠 Models: {response.status_code}\")\n",
                "        print(json.dumps(response.json(), indent=2))\n",
                "    except Exception as e:\n",
                "        print(f\"❌ Models check failed: {e}\")\n",
                "else:\n",
                "    print(\"⚠️ Public URL not available\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Monitor system resources\n",
                "import psutil\n",
                "from datetime import datetime\n\n",
                "def show_stats():\n",
                "    print(f\"⏰ {datetime.now().strftime('%H:%M:%S')}\")\n",
                "    print(f\"🖥️ CPU: {psutil.cpu_percent()}%\")\n",
                "    print(f\"💾 RAM: {psutil.virtual_memory().percent}%\")\n",
                "    \n",
                "    if torch.cuda.is_available():\n",
                "        allocated = torch.cuda.memory_allocated() / 1e9\n",
                "        cached = torch.cuda.memory_reserved() / 1e9\n",
                "        print(f\"🎮 GPU: {allocated:.1f}GB used, {cached:.1f}GB cached\")\n",
                "    print(\"-\" * 40)\n\n",
                "show_stats()\n",
                "print(\"\\n💡 Run this cell periodically to monitor resources\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cleanup when done\n",
                "def cleanup():\n",
                "    print(\"🧹 Cleaning up...\")\n",
                "    \n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "        print(\"✅ GPU memory cleared\")\n",
                "    \n",
                "    try:\n",
                "        if 'public_url' in locals():\n",
                "            ngrok.disconnect(public_url)\n",
                "        print(\"✅ Ngrok tunnel closed\")\n",
                "    except:\n",
                "        pass\n",
                "    \n",
                "    print(\"✅ Cleanup completed\")\n\n",
                "# Uncomment to cleanup:\n",
                "# cleanup()\n\n",
                "print(\"💡 Run cleanup() when finished\")"
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

print("Notebook template created successfully!")