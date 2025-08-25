# Mango Leaf Disease Detection System

## Overview

This is a Flask-based web application for detecting diseases in mango leaves using deep learning and explainable AI techniques. The system uses a ResNet50-based PyTorch model for classification and provides treatment recommendations for detected diseases. It features both single image analysis and batch processing capabilities, with explainable AI methods like GradCAM for visual interpretations of model predictions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework
- **Flask**: Chosen as the web framework for its simplicity and flexibility in handling machine learning applications
- **SQLAlchemy**: Used for database operations with a declarative base model approach
- **ProxyFix**: Implemented for proper handling of proxy headers in production deployments

### Machine Learning Architecture
- **PyTorch**: Core deep learning framework for model inference and training
- **ResNet50 Backbone**: Pre-trained ResNet50 model enhanced with custom classifier layers featuring:
  - Dropout layers for regularization (rates: 0.5, 0.35, 0.25)
  - Batch normalization for training stability
  - Progressive layer size reduction (2048 → 1024 → 512 → 8 classes)
- **Custom MangoLeafModel**: Extends ResNet50 with enhanced classification head for 8 disease classes

### Explainable AI Integration
- **GradCAM Implementation**: Provides visual explanations by highlighting image regions important for classification decisions
- **Hook-based Architecture**: Uses PyTorch forward and backward hooks on the last convolutional layer (layer4[2].conv3) for gradient capture
- **XAI Explainer Class**: Modular design for generating heatmaps and visual explanations

### Data Storage
- **SQLite/PostgreSQL**: Flexible database configuration supporting both development (SQLite) and production (PostgreSQL via DATABASE_URL)
- **Two-table Schema**:
  - `AnalysisResult`: Stores individual image analysis results with predictions, confidence scores, and probability distributions
  - `BatchProcess`: Tracks batch processing jobs with status monitoring and success rate calculations

### File Handling
- **Secure Upload System**: Implements file type validation, size limits (50MB), and secure filename handling
- **Multi-format Support**: Accepts PNG, JPG, JPEG, GIF, and BMP image formats
- **Organized Storage**: Separate directories for uploads and model files

### Treatment Recommendation System
- **Disease Knowledge Base**: Comprehensive treatment database covering multiple disease types (Anthracnose, Bacterial Canker, etc.)
- **Multi-tier Treatment Approach**: 
  - Immediate actions for quick response
  - Chemical treatments with specific product recommendations
  - Organic alternatives for sustainable farming
  - Preventive measures for long-term management

### Frontend Architecture
- **Bootstrap 5**: Responsive UI framework with dark theme support
- **Progressive Enhancement**: JavaScript handles drag-and-drop uploads, real-time previews, and dynamic form interactions
- **Chart.js Integration**: For visualizing confidence scores and probability distributions
- **Font Awesome Icons**: Consistent iconography throughout the interface

### Batch Processing Design
- **Asynchronous Processing**: Supports multiple image uploads with progress tracking
- **Status Management**: Real-time status updates (pending, processing, completed, failed)
- **Grid-based Preview**: Efficient display of multiple uploaded images
- **Success Rate Calculation**: Automatic computation of batch processing success metrics

## External Dependencies

### Machine Learning Libraries
- **PyTorch + TorchVision**: Core deep learning framework and computer vision utilities
- **PIL (Pillow)**: Image processing and format conversion
- **NumPy**: Numerical computations and array operations
- **OpenCV**: Advanced image processing for preprocessing and augmentation

### Web Development Stack
- **Flask Framework**: Web application foundation
- **Flask-SQLAlchemy**: Database ORM integration
- **Werkzeug**: WSGI utilities and secure file handling

### Visualization and UI
- **Matplotlib**: Backend for generating explanation heatmaps and visualizations
- **Chart.js**: Frontend charting library for confidence score displays
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements

### Data Processing
- **Base64**: Image encoding for web display
- **JSON**: Data serialization for probability distributions
- **UUID**: Unique identifier generation for file management

### Development and Deployment
- **Python 3.8+**: Runtime environment
- **CUDA Support**: Optional GPU acceleration for model inference
- **Environment Variables**: Configuration management for database URLs and secrets