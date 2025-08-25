# 🚀 Google Colab GPU Backend Setup Guide

This guide will help you connect your Mango Leaf Disease Detection app to Google Colab for GPU-powered processing.

## 🎯 What This Setup Provides

- **🔥 GPU-Accelerated Inference**: Lightning-fast disease detection using Colab's free GPUs
- **🎨 Background Removal**: AI-powered background removal using RemBG
- **🔍 Explainable AI**: GradCAM, Integrated Gradients, and LIME explanations
- **💰 Cost-Effective**: Use Colab's free GPU instead of expensive cloud instances
- **🌐 Public API**: Access from anywhere via ngrok tunnels

## 📋 Prerequisites

1. **Google Account**: For Google Colab access
2. **Colab Pro/Plus** (Recommended): For longer runtimes and better GPUs
3. **ngrok Account**: For creating public tunnels (free tier available)

## 🔧 Setup Instructions

### Step 1: Prepare Google Colab

1. **Open Google Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Sign in with your Google account

2. **Create New Notebook**
   - Click "New Notebook"
   - Or upload the provided `Mango_Disease_Detection_Colab_Backend.ipynb`

3. **Enable GPU Runtime**
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Choose `T4 GPU` (free) or `A100/V100` (Pro/Plus)
   - Click `Save`

### Step 2: Set Up ngrok (Public Tunnel)

1. **Get ngrok Token**
   - Go to [ngrok.com](https://ngrok.com) and create free account
   - Navigate to [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Copy your authtoken

2. **Configure ngrok in Colab**
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")  # Replace with your token
   ```

### Step 3: Upload Backend Service to Colab

**Option A: Direct Upload**
1. Upload `colab_backend.py` to your Colab notebook
2. Run the installation and setup cells

**Option B: GitHub Integration** (Recommended)
1. Upload your code to GitHub
2. Clone in Colab:
   ```python
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

### Step 4: Upload Your Trained Models

1. **Prepare Model Files**
   - Ensure models are in PyTorch format (`.pth`, `.pth.tar`, `.pt`)
   - Recommended naming: `resnet50_model.pth`, `efficientnet_b0_model.pth`

2. **Upload to Colab**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload your model files
   ```

3. **Or Use Google Drive** (For large models)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Copy models from Drive to Colab
   !cp /content/drive/MyDrive/models/*.pth /content/models/
   ```

### Step 5: Start the Backend Service

1. **Run Installation Cell**
   ```python
   !pip install -q flask pyngrok rembg[gpu] grad-cam captum lime opencv-python-headless timm
   ```

2. **Start the Service**
   ```python
   # This will start Flask and create ngrok tunnel
   python colab_backend.py
   ```

3. **Get Public URL**
   - The notebook will display your ngrok URL
   - Example: `https://abc123.ngrok.io`
   - **Keep this URL** - you'll need it for your local app

### Step 6: Configure Your Local App

1. **Set Environment Variable**
   ```bash
   export COLAB_BACKEND_URL="https://abc123.ngrok.io"
   ```

2. **Or Configure via Web Interface**
   - Start your local app: `python lightweight_app.py`
   - Go to `http://localhost:5001/configure_backend`
   - Enter your ngrok URL
   - Click "Save Configuration"

3. **Test Connection**
   - The status page will show if backend is connected
   - Visit `http://localhost:5001/status`

## 🔥 Complete Colab Notebook Code

Here's the complete code for your Colab notebook:

```python
# Cell 1: Install packages
!pip install -q flask pyngrok rembg[gpu] grad-cam captum lime opencv-python-headless timm

# Cell 2: Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 3: Setup ngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_AUTHTOKEN_HERE")  # Replace with your token

# Cell 4: Upload backend service
# Either upload colab_backend.py or create it inline
with open('colab_backend.py', 'w') as f:
    f.write('''
# Your backend service code goes here
# (Copy from colab_backend.py)
''')

# Cell 5: Upload models
from google.colab import files
import os

os.makedirs('models', exist_ok=True)
print("Upload your trained models:")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith(('.pth', '.pt', '.pth.tar')):
        os.rename(filename, f'models/{filename}')
        print(f"✅ {filename} uploaded to models/")

# Cell 6: Start service
import subprocess
import threading
import time

def run_backend():
    subprocess.run(["python", "colab_backend.py"])

# Start Flask in background
backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()

time.sleep(5)  # Wait for service to start

# Create public tunnel
public_url = ngrok.connect(5000)
print(f"🌐 Public URL: {public_url}")
print(f"Use this URL in your local app: {public_url}")

# Cell 7: Keep alive (run this to prevent timeout)
import time
while True:
    print("Backend running... (Ctrl+C to stop)")
    time.sleep(300)  # Print every 5 minutes
```

## 🔧 Local App Setup

1. **Install Dependencies**
   ```bash
   pip install flask flask-sqlalchemy pillow requests
   ```

2. **Run Lightweight App**
   ```bash
   python lightweight_app.py
   ```

3. **Access App**
   - Open `http://localhost:5001`
   - Configure backend URL
   - Start analyzing images!

## 🚀 Usage Workflow

1. **Start Colab Backend**
   - Run all cells in your Colab notebook
   - Note the ngrok URL

2. **Configure Local App**
   - Enter ngrok URL in configuration
   - Verify connection is healthy

3. **Analyze Images**
   - Upload mango leaf images
   - Select processing options:
     - ✅ Remove Background
     - 🧠 Choose AI Model (ResNet50, EfficientNet)
     - 🔍 Select Explanation Method (GradCAM, LIME)
   - Get comprehensive results with:
     - Disease prediction
     - Confidence score
     - Visual heatmaps
     - Treatment recommendations

## 📊 Expected Results

Your analysis will include:

- **🎯 Disease Detection**: Accurate classification with confidence scores
- **🖼️ Visual Explanations**: Heatmaps showing AI focus areas
- **🏥 Treatment Advice**: Expert recommendations for detected diseases
- **⚡ Fast Processing**: GPU acceleration for quick results
- **📈 Comprehensive Report**: Downloadable results with all details

## 🛠️ Troubleshooting

### Common Issues

1. **"Backend Not Responding"**
   - Check if Colab notebook is still running
   - Verify ngrok tunnel is active
   - Restart the backend service in Colab

2. **"Model Not Found"**
   - Ensure models are uploaded to `/content/models/`
   - Check file naming convention
   - Verify model format (.pth, .pth.tar)

3. **"GPU Not Available"**
   - Confirm GPU runtime is enabled in Colab
   - Check if GPU quota is exceeded
   - Try reconnecting to runtime

4. **"Ngrok Tunnel Expired"**
   - Free ngrok tunnels have time limits
   - Restart the tunnel creation cell
   - Consider upgrading to ngrok Pro for persistent tunnels

### Performance Tips

1. **Keep Colab Active**
   - Run the keep-alive cell to prevent timeout
   - Interact with notebook periodically
   - Consider Colab Pro for longer runtimes

2. **Optimize Model Loading**
   - Models are cached after first load
   - Use smaller models for faster inference
   - Clear cache if memory issues occur

3. **Background Removal**
   - Can be disabled for faster processing
   - Uses additional GPU memory
   - Skip for images with clean backgrounds

## 💡 Advanced Tips

### Multiple Model Support
```python
# In Colab, you can load multiple models
models = {
    'resnet50': load_model('models/resnet50_model.pth'),
    'efficientnet_b0': load_model('models/efficientnet_b0_model.pth'),
    'densenet121': load_model('models/densenet121_model.pth')
}
```

### Custom Preprocessing
```python
# Modify preprocessing pipeline
def custom_preprocess(image):
    # Add your custom preprocessing steps
    return processed_image
```

### Batch Processing
```python
# Process multiple images at once
def process_batch(images):
    results = []
    for image in images:
        result = process_single_image(image)
        results.append(result)
    return results
```

## 🌐 Deployment Options

### Option 1: Local Development
- Run Flask app locally
- Connect to Colab backend
- Perfect for development and testing

### Option 2: Cloud Deployment
- Deploy Flask app to Heroku/Railway/Render
- Connect to Colab backend
- Production-ready solution

### Option 3: Hybrid Setup
- Frontend on cloud platform
- Backend on Colab Pro for GPU access
- Best of both worlds

## 📈 Monitoring and Maintenance

### Keep Track Of:
- Colab runtime usage
- ngrok tunnel status
- Model performance metrics
- API response times

### Regular Tasks:
- Update model weights
- Monitor GPU usage
- Check for Colab timeouts
- Backup important results

## 🎉 You're Ready!

With this setup, you have:
- ✅ GPU-powered disease detection
- ✅ Professional web interface
- ✅ Explainable AI capabilities
- ✅ Cost-effective solution
- ✅ Scalable architecture

Start analyzing mango leaf diseases with the power of AI! 🥭🔬