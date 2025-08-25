# Google Colab Integration Setup Guide

This guide will help you connect your web app to Google Colab for GPU-powered image processing, including background removal, disease classification, and explainability analysis.

## Overview

The integration allows you to:
- ✅ Remove image backgrounds using GPU-accelerated models
- ✅ Perform disease classification with higher accuracy
- ✅ Generate comprehensive explainability reports (GradCAM, LIME, etc.)
- ✅ Automatically fallback to local processing if Colab is unavailable

## Step 1: Set up Google Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload the notebook**: Upload `colab_inference_server.ipynb` to your Google Drive or open it directly in Colab

3. **Enable GPU**: 
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Choose `T4 GPU` (recommended) or higher if available

4. **Run the notebook**:
   - Execute all cells in order
   - The first cell will install required packages (takes 2-3 minutes)
   - Wait for all installations to complete

5. **Get the ngrok URL**:
   - The last cell will display an ngrok URL like: `https://1234-56-78-90-123.ngrok-free.app`
   - **Copy this URL** - you'll need it for your web app

## Step 2: Configure Your Web App

### Option A: Set Environment Variable (Recommended)

1. **Create .env file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit .env file**:
   ```bash
   # Add your Colab ngrok URL
   COLAB_API_URL=https://1234-56-78-90-123.ngrok-free.app
   ```

3. **Restart your web app**:
   ```bash
   python app.py
   ```

### Option B: Set URL via API (Dynamic)

You can also set the Colab URL dynamically without restarting:

```bash
# Update Colab URL via API
curl -X POST http://localhost:5000/api/colab/config \
  -H "Content-Type: application/json" \
  -d '{"url": "https://1234-56-78-90-123.ngrok-free.app"}'
```

## Step 3: Verify Connection

1. **Check health endpoint**:
   ```bash
   curl http://localhost:5000/api/health
   ```

   Look for:
   ```json
   {
     "colab_service": {
       "is_available": true,
       "url": "https://1234-56-78-90-123.ngrok-free.app"
     },
     "processing_modes": {
       "colab": true,
       "background_removal": true
     }
   }
   ```

2. **Test image processing**:
   - Upload an image through your web app
   - Check the results for `"processing_source": "colab"`
   - Look for background removal indicators

## Step 4: Usage

### Normal Operation
- Images will automatically be processed via Colab when available
- Background removal happens automatically
- Enhanced explainability features are enabled
- Results include processing source information

### Fallback Mode
- If Colab is unavailable, processing falls back to local mode
- Local processing doesn't include background removal
- Basic explainability features still work

## Troubleshooting

### Common Issues

1. **"Colab service is not available"**
   - Check if your Colab notebook is still running
   - Verify the ngrok URL is correct and accessible
   - Try refreshing the Colab notebook

2. **Timeout errors**
   - Large images may take longer to process
   - Colab GPU allocation might be limited
   - Try smaller images or wait and retry

3. **ngrok URL expired**
   - ngrok URLs expire after inactivity
   - Re-run the last cell in Colab to get a new URL
   - Update your .env file or use the API to update the URL

### Monitoring

Check the status anytime:
```bash
# Get current Colab service status
curl http://localhost:5000/api/colab/config

# Full health check
curl http://localhost:5000/api/health
```

### Colab Session Management

**Important Notes:**
- Colab sessions have time limits (varies by plan)
- Free tier: ~12 hours, then requires restart
- Pro/Pro+: Longer sessions, better GPU access
- Keep the Colab notebook running for continuous service

**Best Practices:**
- Monitor your Colab compute units usage
- Restart sessions before they expire
- Use Colab Pro+ for production workloads

## Advanced Configuration

### Custom Timeouts
```python
# In your app, modify colab_service.py
colab_service = ColabService(
    timeout=180,  # 3 minutes for large images
    max_retries=5
)
```

### Batch Processing
The system automatically supports batch processing through Colab:
- Multiple images processed in parallel
- Efficient GPU utilization
- Progress tracking

### Model Updates
To use different models in Colab:
1. Modify the `MangoClassifier` class in the notebook
2. Update the model loading logic
3. Restart the Colab session

## Security Considerations

- ngrok URLs are publicly accessible
- Consider using ngrok authentication for production
- Monitor your Colab usage to prevent abuse
- Don't commit actual URLs to version control

## Cost Optimization

- Use Colab Pro+ for consistent GPU access
- Monitor compute unit usage
- Consider local processing for development
- Use batch processing for multiple images

---

## Quick Start Commands

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your Colab URL

# 2. Start web app
python app.py

# 3. Test connection
curl http://localhost:5000/api/health

# 4. Upload test image and verify Colab processing
```

For issues or questions, check the logs in both your web app and Colab notebook.