# Mango Leaf Disease Detection - Explainable AI System

This enhanced version of the Mango Leaf Disease Detection app now includes comprehensive explainability features that allow users to understand how AI models make their decisions.

## 🆕 New Features

### 1. **Comprehensive Explainability Analysis**
- **Multiple CAM Methods**: GradCAM, HiResCAM, ScoreCAM, GradCAM++, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, GradCAMElementWise
- **Advanced Methods**: Integrated Gradients, LIME (Local Interpretable Model-agnostic Explanations)
- **Visual Heatmaps**: See exactly which parts of the leaf the AI focuses on for disease detection

### 2. **Multi-Model Architecture Support**
- **Multiple Models**: ResNet50, ResNet101, DenseNet121, EfficientNet-B0, EfficientNet-B3, Vision Transformer (ViT), ConvNeXt
- **Model Comparison**: Compare predictions across different architectures
- **Performance Analytics**: View accuracy, confidence, and processing time for each model

### 3. **Interactive Web Interface**
- **Model Selection**: Choose which models to use for analysis
- **Method Selection**: Select specific explainability methods
- **Real-time Results**: View explanations as they're generated
- **Export Functionality**: Download results in various formats

## 🚀 Quick Start

### Prerequisites
Make sure you have the following dependencies installed:
```bash
pip install grad-cam captum lime timm
```

Or use the provided `pyproject.toml`:
```bash
pip install -e .
```

### Running the Application
1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Navigate to the "Explainability" section from the main menu

## 📋 Usage Guide

### Basic Explainability Analysis

1. **Upload Image**: 
   - Go to `/explainability`
   - Upload a mango leaf image
   - The system supports JPG, PNG, GIF, BMP formats

2. **Select Models**:
   - Choose from available trained models
   - Each model shows accuracy and file size information
   - Select multiple models for comparison

3. **Choose Methods**:
   - **Visual Attribution**: GradCAM, HiResCAM, ScoreCAM, etc.
   - **Advanced Methods**: Integrated Gradients, LIME
   - Use "Recommended" for quick setup

4. **Start Analysis**:
   - Click "Start Explainability Analysis"
   - View real-time progress
   - Results display automatically

### Model Comparison

1. **Access Model Comparison**:
   - Navigate to `/model_comparison`
   - Upload an image for analysis

2. **View Results**:
   - See predictions from all available models
   - Compare confidence scores
   - View consensus predictions
   - Export comparison results

### Understanding Results

#### Explainability Visualizations
- **Red/Hot Areas**: Regions the model considers important for the prediction
- **Blue/Cool Areas**: Regions the model considers less relevant
- **Intensity**: Stronger colors indicate higher importance

#### Method Descriptions
- **GradCAM**: Shows where the model looks using gradient information
- **ScoreCAM**: Uses forward passes to generate importance maps
- **Integrated Gradients**: Shows pixel-level importance through path integration
- **LIME**: Explains predictions by learning local interpretable models

## 🏗️ Architecture Overview

### Core Components

1. **ComprehensiveXAIExplainer** (`comprehensive_xai_explainer.py`)
   - Main explainability engine
   - Supports all CAM methods and advanced techniques
   - Handles multiple model architectures

2. **MultiModelManager** (`multi_model_manager.py`)
   - Manages multiple trained models
   - Handles model loading and metadata
   - Provides model comparison functionality

3. **Enhanced Flask Routes** (`app.py`)
   - `/explainability` - Main explainability interface
   - `/model_comparison` - Model comparison interface
   - `/api/explainability_analysis` - API for explainability analysis
   - `/api/model_comparison` - API for model comparison
   - `/api/models` - API for model information

### File Structure
```
├── app.py                           # Main Flask application
├── comprehensive_xai_explainer.py   # Explainability engine
├── multi_model_manager.py          # Multi-model management
├── templates/
│   ├── base.html                   # Base template with enhanced UI
│   ├── explainability.html        # Explainability interface
│   ├── model_comparison.html      # Model comparison interface
│   ├── index.html                 # Enhanced homepage
│   ├── results.html               # Results display
│   ├── batch.html                 # Batch processing
│   └── history.html               # Analysis history
└── models/                        # Directory for trained models
```

## 🔧 Configuration

### Model Directory
Place your trained models in the `models/` directory. Supported formats:
- `.pth` - PyTorch model files
- `.pt` - PyTorch model files  
- `.pth.tar` - PyTorch checkpoint files

### Model Naming Convention
For automatic architecture detection, include the architecture name in the filename:
- `resnet50_model.pth`
- `efficientnet_b0_best.pth`
- `vit_base_trained.pth`

### Explainability Methods Configuration

You can customize which methods are available by modifying the `cam_methods` dictionary in `ComprehensiveXAIExplainer`:

```python
self.cam_methods = {
    'GradCAM': GradCAM,
    'HiResCAM': HiResCAM,
    'ScoreCAM': ScoreCAM,
    # Add or remove methods as needed
}
```

## 📊 Performance Considerations

### Method Performance
- **Fastest**: GradCAM, HiResCAM
- **Medium**: GradCAM++, XGradCAM, EigenCAM
- **Slower**: ScoreCAM, AblationCAM, FullGrad
- **Slowest**: LIME, Integrated Gradients

### Memory Usage
- Each loaded model consumes GPU/CPU memory
- Unload unused models to free memory
- Use model manager's `unload_model()` function

### Recommendations
- Start with 1-2 models and 3-4 methods for testing
- Use "Recommended" methods for balanced performance
- Consider using GPU for faster processing

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - Ensure models are in the `models/` directory
   - Check file permissions
   - Verify model file format

2. **"Method failed" error**:
   - Some methods may not work with all architectures
   - Try different explainability methods
   - Check GPU memory availability

3. **Slow processing**:
   - Reduce number of selected methods
   - Use faster methods (GradCAM, HiResCAM)
   - Consider using CPU if GPU memory is limited

4. **Import errors**:
   - Install required dependencies: `pip install grad-cam captum lime timm`
   - Check PyTorch compatibility

### Debug Mode
Enable debug logging in `app.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Adding New Explainability Methods
1. Add the method to `comprehensive_xai_explainer.py`
2. Update the `cam_methods` dictionary
3. Add method description in the template
4. Test with different model architectures

### Adding New Model Architectures
1. Update `AdvancedMangoLeafModel` in `multi_model_manager.py`
2. Add architecture-specific target layer detection
3. Test explainability methods with the new architecture

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Grad-CAM**: For comprehensive CAM implementations
- **Captum**: For advanced attribution methods
- **LIME**: For model-agnostic explanations
- **Timm**: For additional model architectures

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the debug logs
3. Open an issue on the project repository

---

**Happy Explaining! 🔍🌿**