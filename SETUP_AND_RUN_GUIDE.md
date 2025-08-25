# 🥭 Mango Leaf Disease Detection - Complete Setup and Run Guide

This comprehensive guide will walk you through every step needed to set up and run the Mango Leaf Disease Detection web application.

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Running the Application](#running-the-application)
5. [Deployment Options](#deployment-options)
6. [Using the Web Interface](#using-the-web-interface)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

---

## 🖥️ System Requirements

### Minimum Requirements:
- **Operating System:** Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python:** 3.11 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 2GB free space
- **Internet:** Required for initial setup and model downloads

### Recommended for AI Processing:
- **RAM:** 8GB or more
- **GPU:** NVIDIA GPU with CUDA support (optional but faster)
- **CPU:** Multi-core processor (4+ cores recommended)

---

## 🔧 Installation Methods

### Method 1: Using pip (Recommended)
```bash
pip install -r requirements.txt
```

### Method 2: Using uv (Faster)
```bash
# Install uv first
pip install uv
# Install dependencies
uv sync
```

### Method 3: Using conda
```bash
conda create -n mango-detection python=3.11
conda activate mango-detection
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Setup

### Step 1: Download/Clone the Project
If you haven't already, get the project files:
```bash
# If using git
git clone <repository-url>
cd mango-disease-detection

# Or download and extract the ZIP file
```

### Step 2: Verify Python Installation
```bash
# Check Python version (must be 3.11+)
python3 --version
# Expected output: Python 3.11.x or higher

# On Windows, you might need to use:
python --version
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv mango_env

# Activate it
# On Linux/macOS:
source mango_env/bin/activate
# On Windows:
mango_env\Scripts\activate

# You should see (mango_env) in your terminal prompt
```

### Step 4: Install Dependencies
```bash
# Make sure you're in the project directory
cd /path/to/mango-disease-detection

# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt

# Wait for installation to complete (this may take 5-10 minutes)
```

### Step 5: Verify Installation
```bash
# Test if key packages are installed
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import flask; print('Flask:', flask.__version__)"
python3 -c "import cv2; print('OpenCV: OK')"
python3 -c "from pytorch_grad_cam import GradCAM; print('GradCAM: OK')"
```

### Step 6: Create Required Directories
```bash
# Create directories if they don't exist
mkdir -p uploads
mkdir -p models
mkdir -p static/results
mkdir -p logs

# Verify directories were created
ls -la
# You should see: uploads/, models/, static/, templates/, etc.
```

### Step 7: Check File Permissions
```bash
# Make main files executable (Linux/macOS)
chmod +x main.py
chmod +x quick_start.py

# Verify
ls -la *.py
```

---

## 🌐 Running the Application

### Option A: Quick Start (Easiest)
```bash
# Navigate to project directory
cd /path/to/mango-disease-detection

# Activate virtual environment (if created)
source mango_env/bin/activate  # Linux/macOS
# or
mango_env\Scripts\activate     # Windows

# Run the application
python3 main.py
```

**Expected Output:**
```
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[your-ip]:5000
 * Press CTRL+C to quit
```

### Option B: Production Mode
```bash
# Install gunicorn (if not already installed)
pip install gunicorn

# Run with gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 main:app
```

### Option C: Lightweight Mode (Frontend Only)
```bash
# Run lightweight frontend (connects to Colab backend)
python3 lightweight_app.py
```

**Access Points:**
- **Main App:** http://localhost:5000
- **Lightweight App:** http://localhost:5001

---

## 📱 Deployment Options

### Local Development
```bash
# Simple local run
python3 main.py

# Access at: http://localhost:5000
```

### Network Access
```bash
# Allow access from other devices on your network
python3 main.py
# or
gunicorn --bind 0.0.0.0:5000 main:app

# Access at: http://YOUR_IP_ADDRESS:5000
# Find your IP: ip addr show (Linux) or ipconfig (Windows)
```

### Google Colab Integration

#### Step 1: Set up Colab Backend
1. Open Google Colab (https://colab.research.google.com/)
2. Create a new notebook
3. Copy the contents of `colab_backend.py` to the notebook
4. Run all cells
5. Note the ngrok URL provided (e.g., https://abc123.ngrok.io)

#### Step 2: Configure Local Frontend
```bash
# Edit the configuration to point to Colab
# In lightweight_app.py, update the COLAB_BACKEND_URL
export COLAB_BACKEND_URL="https://your-ngrok-url.ngrok.io"

# Run lightweight frontend
python3 lightweight_app.py
```

### Cloud Deployment (Heroku Example)
```bash
# Install Heroku CLI first, then:
heroku create your-app-name
git push heroku main
heroku open
```

---

## 🎮 Using the Web Interface

### Step 1: Access the Application
1. Open your web browser
2. Go to: http://localhost:5000
3. You should see the Mango Disease Detection homepage

### Step 2: Upload an Image
1. Click on "Upload Image" or "Single Image Analysis"
2. Select a mango leaf image file (JPG, PNG, etc.)
3. Click "Upload" or "Analyze"

### Step 3: View Results
1. Wait for processing (usually 2-10 seconds)
2. View the disease prediction
3. Examine the confidence scores
4. Check the explainable AI visualizations (heatmaps)
5. Read treatment recommendations

### Step 4: Batch Processing (Optional)
1. Go to "Batch Processing" section
2. Upload multiple images (ZIP file or select multiple files)
3. Wait for processing
4. Download results as CSV or view in browser

### Features Available:
- **Disease Classification:** AI identifies mango leaf diseases
- **Confidence Scores:** Shows how certain the AI is
- **Explainable AI:** Visual heatmaps showing what the AI focused on
- **Treatment Advice:** Expert recommendations for each disease
- **History:** View past analyses
- **Model Comparison:** Compare different AI models

---

## 🚨 Troubleshooting

### Problem 1: "Command not found: python3"
**Solution:**
```bash
# Try using 'python' instead of 'python3'
python main.py

# Or install Python 3.11+
# Windows: Download from python.org
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11
```

### Problem 2: "ModuleNotFoundError"
**Solution:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or try installing individually:
pip install torch torchvision flask grad-cam pillow opencv-python
```

### Problem 3: "Port 5000 is already in use"
**Solution:**
```bash
# Find and kill the process
lsof -ti:5000 | xargs kill -9  # Linux/macOS
netstat -ano | findstr :5000   # Windows (then kill the PID)

# Or use a different port
python3 main.py --port 8080
```

### Problem 4: "No module named 'torch'"
**Solution:**
```bash
# Install PyTorch
pip install torch torchvision torchaudio

# For GPU support (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem 5: "Permission denied"
**Solution:**
```bash
# Fix file permissions (Linux/macOS)
chmod 755 main.py
chmod -R 755 uploads/

# Run as administrator (Windows)
# Right-click Command Prompt → "Run as administrator"
```

### Problem 6: "Database is locked"
**Solution:**
```bash
# Reset the database
rm mango_disease.db
rm mango_disease_lightweight.db

# Restart the application
python3 main.py
```

### Problem 7: Slow Performance
**Solutions:**
```bash
# Use GPU acceleration (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Reduce image size in uploads/
# Use lighter model (modify ml_models.py)
# Use Colab backend for heavy processing
```

### Problem 8: "Import Error: No module named 'cv2'"
**Solution:**
```bash
pip install opencv-python
# or
pip install opencv-python-headless
```

---

## ⚙️ Advanced Configuration

### Environment Variables
```bash
# Create .env file for configuration
echo "DATABASE_URL=sqlite:///mango_disease.db" > .env
echo "SESSION_SECRET=your-secret-key-here" >> .env
echo "DEBUG=True" >> .env
echo "COLAB_BACKEND_URL=http://localhost:5000" >> .env
```

### Custom Model Path
```bash
# Place your trained models in models/ directory
cp your_model.pth models/
# Update ml_models.py to load your model
```

### Database Configuration
```bash
# For PostgreSQL (production)
export DATABASE_URL="postgresql://user:password@localhost/mango_db"

# For MySQL
export DATABASE_URL="mysql://user:password@localhost/mango_db"
```

### Logging Configuration
```bash
# Create logs directory
mkdir -p logs

# Enable detailed logging (edit app.py)
# Change: logging.basicConfig(level=logging.INFO)
# To: logging.basicConfig(level=logging.DEBUG)
```

---

## 🔍 Testing Your Setup

### Quick Test
```bash
# Test the application
curl http://localhost:5000
# Should return HTML of the homepage

# Test API endpoint
curl -X POST http://localhost:5000/api/health
# Should return {"status": "healthy"}
```

### Full Test with Image
1. Save a test image in `uploads/test.jpg`
2. Open browser to http://localhost:5000
3. Upload the test image
4. Verify you get predictions and visualizations

---

## 📞 Getting Help

### Check Logs
```bash
# View application logs
tail -f logs/app.log

# Check for errors
grep ERROR logs/app.log
```

### Common Commands Reference
```bash
# Start app
python3 main.py

# Stop app
Ctrl+C

# Restart app
Ctrl+C then python3 main.py

# Check processes
ps aux | grep python

# Check ports
netstat -tulpn | grep :5000
```

### Debug Mode
```bash
# Run with extra debugging
export FLASK_DEBUG=1
python3 main.py
```

---

## ✅ Success Checklist

- [ ] Python 3.11+ installed and verified
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] Required directories created
- [ ] Application starts without errors
- [ ] Can access web interface at http://localhost:5000
- [ ] Can upload and analyze test images
- [ ] AI predictions and visualizations work
- [ ] Treatment recommendations display correctly

**🎉 Congratulations! Your Mango Leaf Disease Detection system is now running!**

---

## 📚 Additional Resources

- **Project Documentation:** See other .md files in this directory
- **Model Training:** Refer to ml_models.py for model architecture
- **API Documentation:** Check app.py for available endpoints
- **Troubleshooting:** See issue tracker or logs/ directory

For more help, check the application logs in the `logs/` directory or review the error messages in your terminal.