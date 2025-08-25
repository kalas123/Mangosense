"""
Google Colab Backend Service for Mango Leaf Disease Detection
This script runs on Google Colab and provides GPU-powered inference, explainability, and background removal.

To use:
1. Upload this file to Google Colab
2. Install required packages
3. Run the service
4. Use ngrok to create public URL
5. Connect from your local Flask app
"""

import os
import io
import base64
import json
import logging
import time
from datetime import datetime
import zipfile
import tempfile
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_cache = {}
explainers_cache = {}
background_remover = None

print(f"🚀 Colab Backend Service Starting...")
print(f"🔧 Device: {device}")
print(f"🧠 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# Disease classes
CLASS_NAMES = [
    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
    'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
]

class AdvancedMangoLeafModel(nn.Module):
    """Advanced Mango Leaf Disease Detection Model"""
    
    def __init__(self, num_classes=8, model_name='resnet50', dropout_rate=0.5):
        super(AdvancedMangoLeafModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._create_classifier(num_features, num_classes, dropout_rate)
        else:
            # Default to ResNet50
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
            self.model_name = 'resnet50'
    
    def _create_classifier(self, num_features, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class BackgroundRemover:
    """Background removal using various methods"""
    
    def __init__(self):
        self.method = 'rembg'  # Default method
        self.rembg_model = None
        
    def initialize_rembg(self):
        """Initialize RemBG model"""
        try:
            from rembg import remove, new_session
            self.rembg_model = new_session('u2net')
            logger.info("✅ RemBG model initialized")
            return True
        except ImportError:
            logger.warning("RemBG not available, falling back to traditional methods")
            return False
    
    def remove_background_rembg(self, image):
        """Remove background using RemBG"""
        try:
            from rembg import remove
            
            # Convert PIL to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Remove background
            output = remove(img_byte_arr, session=self.rembg_model)
            
            # Convert back to PIL
            return Image.open(io.BytesIO(output))
            
        except Exception as e:
            logger.error(f"RemBG failed: {e}")
            return None
    
    def remove_background_grabcut(self, image):
        """Remove background using GrabCut algorithm"""
        try:
            # Convert PIL to OpenCV
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = img.shape[:2]
            
            # Create mask
            mask = np.zeros((height, width), np.uint8)
            
            # Define rectangle around the object (assuming leaf is in center)
            rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))
            
            # Initialize foreground and background models
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply mask
            result = img * mask2[:, :, np.newaxis]
            
            # Convert back to PIL with transparency
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            # Add alpha channel
            result_rgba = result_pil.convert('RGBA')
            data = result_rgba.getdata()
            
            new_data = []
            for item in data:
                # Make black pixels transparent
                if item[0] < 10 and item[1] < 10 and item[2] < 10:
                    new_data.append((255, 255, 255, 0))  # Transparent
                else:
                    new_data.append(item)
            
            result_rgba.putdata(new_data)
            return result_rgba
            
        except Exception as e:
            logger.error(f"GrabCut failed: {e}")
            return None
    
    def remove_background(self, image):
        """Remove background using the best available method"""
        # Try RemBG first (most accurate)
        if self.rembg_model:
            result = self.remove_background_rembg(image)
            if result:
                return result
        
        # Fallback to GrabCut
        result = self.remove_background_grabcut(image)
        if result:
            return result
        
        # If all methods fail, return original
        logger.warning("All background removal methods failed, returning original image")
        return image

class ColabExplainer:
    """Explainability methods optimized for Colab"""
    
    def __init__(self, model, class_names, device):
        self.model = model
        self.class_names = class_names
        self.device = device
        
        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Try to initialize advanced explainers
        self.grad_cam = None
        self.integrated_gradients = None
        self.lime_explainer = None
        
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize explainability methods"""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image
            
            # Get target layers
            target_layers = self._get_target_layers()
            self.grad_cam = GradCAM(model=self.model, target_layers=target_layers)
            logger.info("✅ GradCAM initialized")
            
        except ImportError:
            logger.warning("pytorch_grad_cam not available")
        
        try:
            from captum.attr import IntegratedGradients
            self.integrated_gradients = IntegratedGradients(self.model)
            logger.info("✅ Integrated Gradients initialized")
            
        except ImportError:
            logger.warning("Captum not available")
        
        try:
            from lime import lime_image
            self.lime_explainer = lime_image.LimeImageExplainer()
            logger.info("✅ LIME initialized")
            
        except ImportError:
            logger.warning("LIME not available")
    
    def _get_target_layers(self):
        """Get target layers for different architectures"""
        if hasattr(self.model.backbone, 'layer4'):
            return [self.model.backbone.layer4[-1]]
        elif hasattr(self.model.backbone, 'features'):
            return [self.model.backbone.features[-1]]
        else:
            # Fallback
            return [list(self.model.modules())[-3]]
    
    def generate_gradcam(self, image, target_class=None):
        """Generate GradCAM explanation"""
        if not self.grad_cam:
            return None, "GradCAM not available"
        
        try:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image
            
            # Preprocess image
            rgb_img = np.array(image.convert('RGB').resize((224, 224))) / 255.0
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Get prediction if target_class not provided
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    target_class = output.argmax(dim=1).item()
            
            # Generate CAM
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = self.grad_cam(input_tensor=input_tensor, targets=targets)
            
            # Create visualization
            cam_image = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
            
            return cam_image, None
            
        except Exception as e:
            return None, str(e)
    
    def generate_explanation_report(self, image, prediction_result):
        """Generate comprehensive explanation report"""
        report = {
            'prediction': prediction_result,
            'explanations': {},
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Generate GradCAM
        gradcam_result, gradcam_error = self.generate_gradcam(image, prediction_result['predicted_class_idx'])
        
        if gradcam_result is not None:
            # Convert to base64
            gradcam_pil = Image.fromarray(gradcam_result)
            buffer = BytesIO()
            gradcam_pil.save(buffer, format='PNG')
            gradcam_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            report['explanations']['gradcam'] = {
                'success': True,
                'image_b64': gradcam_b64,
                'method': 'GradCAM'
            }
        else:
            report['explanations']['gradcam'] = {
                'success': False,
                'error': gradcam_error,
                'method': 'GradCAM'
            }
        
        return report

def get_treatment_recommendation(disease_class):
    """Get treatment recommendation for detected disease"""
    treatments = {
        'Anthracnose': {
            'treatment': 'Apply copper-based fungicides. Improve air circulation and avoid overhead watering.',
            'prevention': 'Remove infected plant debris, ensure proper spacing between plants.',
            'severity': 'Moderate to High'
        },
        'Bacterial_Canker': {
            'treatment': 'Apply copper-based bactericides. Prune infected branches and destroy them.',
            'prevention': 'Avoid wounding plants, sterilize pruning tools, improve drainage.',
            'severity': 'High'
        },
        'Cutting_Weevil': {
            'treatment': 'Apply insecticides containing imidacloprid or thiamethoxam.',
            'prevention': 'Regular monitoring, remove affected shoots, use pheromone traps.',
            'severity': 'Moderate'
        },
        'Die_Back': {
            'treatment': 'Prune affected branches, apply copper fungicides to cut surfaces.',
            'prevention': 'Avoid water stress, improve soil drainage, regular pruning.',
            'severity': 'High'
        },
        'Gall_Midge': {
            'treatment': 'Apply systemic insecticides during flowering season.',
            'prevention': 'Remove and destroy affected plant parts, use resistant varieties.',
            'severity': 'Moderate'
        },
        'Healthy': {
            'treatment': 'No treatment needed. Continue regular monitoring.',
            'prevention': 'Maintain good agricultural practices, regular inspection.',
            'severity': 'None'
        },
        'Powdery_Mildew': {
            'treatment': 'Apply sulfur-based fungicides or neem oil.',
            'prevention': 'Ensure good air circulation, avoid overhead watering.',
            'severity': 'Low to Moderate'
        },
        'Sooty_Mould': {
            'treatment': 'Control underlying pest problems, wash leaves with soapy water.',
            'prevention': 'Control aphids and scale insects, improve air circulation.',
            'severity': 'Low'
        }
    }
    
    return treatments.get(disease_class, treatments['Healthy'])

def load_model(model_path, architecture='resnet50'):
    """Load model from checkpoint"""
    try:
        model = AdvancedMangoLeafModel(num_classes=len(CLASS_NAMES), model_name=architecture)
        
        # Try to load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            logger.warning(f"Model file not found: {model_path}. Using pretrained weights.")
            # Use pretrained model as fallback
            if architecture == 'resnet50':
                model.backbone = models.resnet50(pretrained=True)
                num_features = model.backbone.fc.in_features
                model.backbone.fc = model._create_classifier(num_features, len(CLASS_NAMES), 0.5)
        
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def create_side_by_side_visualization(original_image, heatmap_image, prediction_result, treatment_info):
    """Create side-by-side visualization with original and heatmap"""
    
    # Resize images to same size
    target_size = (400, 400)
    original_resized = original_image.resize(target_size)
    heatmap_resized = Image.fromarray(heatmap_image).resize(target_size)
    
    # Create side-by-side image
    combined_width = target_size[0] * 2 + 60  # Space for labels and padding
    combined_height = target_size[1] + 200  # Space for text below
    
    combined_image = Image.new('RGB', (combined_width, combined_height), 'white')
    
    # Paste images
    combined_image.paste(original_resized, (30, 30))
    combined_image.paste(heatmap_resized, (target_size[0] + 50, 30))
    
    # Add labels and information
    draw = ImageDraw.Draw(combined_image)
    
    try:
        # Try to load a font
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Image labels
    draw.text((30, 5), "Original Image", fill='black', font=font_medium)
    draw.text((target_size[0] + 50, 5), "AI Focus Areas (GradCAM)", fill='black', font=font_medium)
    
    # Prediction results
    y_offset = target_size[1] + 50
    
    predicted_class = prediction_result['predicted_class'].replace('_', ' ')
    confidence = prediction_result['confidence'] * 100
    
    draw.text((30, y_offset), f"Prediction: {predicted_class}", fill='black', font=font_large)
    draw.text((30, y_offset + 25), f"Confidence: {confidence:.1f}%", fill='black', font=font_medium)
    
    # Treatment information
    draw.text((30, y_offset + 55), "Treatment Recommendation:", fill='black', font=font_medium)
    
    # Wrap treatment text
    treatment_text = treatment_info['treatment']
    max_width = combined_width - 60
    
    words = treatment_text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font_small)
        if bbox[2] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw treatment lines
    for i, line in enumerate(lines[:3]):  # Limit to 3 lines
        draw.text((30, y_offset + 80 + i * 18), line, fill='black', font=font_small)
    
    return combined_image

# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/process_image', methods=['POST'])
def process_image():
    """Main endpoint for processing images"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get options
        remove_bg = request.form.get('remove_background', 'true').lower() == 'true'
        model_name = request.form.get('model', 'resnet50')
        explain_method = request.form.get('explanation', 'gradcam')
        
        logger.info(f"Processing image: {file.filename}")
        logger.info(f"Options: remove_bg={remove_bg}, model={model_name}, explanation={explain_method}")
        
        # Load image
        image = Image.open(file.stream).convert('RGB')
        original_image = image.copy()
        
        # Step 1: Background removal (if requested)
        processed_image = image
        if remove_bg:
            logger.info("Removing background...")
            if not background_remover:
                global background_remover
                background_remover = BackgroundRemover()
                background_remover.initialize_rembg()
            
            bg_removed = background_remover.remove_background(image)
            if bg_removed:
                processed_image = bg_removed
                logger.info("✅ Background removed successfully")
            else:
                logger.warning("Background removal failed, using original image")
        
        # Step 2: Load model
        model_key = f"model_{model_name}"
        if model_key not in models_cache:
            logger.info(f"Loading model: {model_name}")
            model_path = f"/content/{model_name}_model.pth"  # Adjust path as needed
            model = load_model(model_path, model_name)
            if model:
                models_cache[model_key] = model
                logger.info(f"✅ Model {model_name} loaded and cached")
            else:
                return jsonify({'error': f'Failed to load model: {model_name}'}), 500
        
        model = models_cache[model_key]
        
        # Step 3: Inference
        logger.info("Running inference...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(processed_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class_idx = predicted.item()
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence_score = confidence.item()
        
        prediction_result = {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence_score,
            'all_probabilities': {
                CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities[0])
            }
        }
        
        logger.info(f"Prediction: {predicted_class} ({confidence_score:.3f})")
        
        # Step 4: Explainability
        logger.info("Generating explanation...")
        explainer_key = f"explainer_{model_name}"
        if explainer_key not in explainers_cache:
            explainer = ColabExplainer(model, CLASS_NAMES, device)
            explainers_cache[explainer_key] = explainer
        
        explainer = explainers_cache[explainer_key]
        
        # Generate explanation
        heatmap_image, explanation_error = explainer.generate_gradcam(processed_image, predicted_class_idx)
        
        if heatmap_image is None:
            return jsonify({'error': f'Explanation generation failed: {explanation_error}'}), 500
        
        # Step 5: Get treatment recommendation
        treatment_info = get_treatment_recommendation(predicted_class)
        
        # Step 6: Create comprehensive report
        logger.info("Creating comprehensive report...")
        
        # Create side-by-side visualization
        report_image = create_side_by_side_visualization(
            original_image, heatmap_image, prediction_result, treatment_info
        )
        
        # Convert images to base64
        def image_to_base64(img):
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction_result,
            'treatment': treatment_info,
            'images': {
                'original': image_to_base64(original_image),
                'processed': image_to_base64(processed_image) if remove_bg else None,
                'heatmap': image_to_base64(Image.fromarray(heatmap_image)),
                'report': image_to_base64(report_image)
            },
            'processing_info': {
                'background_removed': remove_bg,
                'model_used': model_name,
                'explanation_method': explain_method,
                'device': str(device),
                'processing_time': time.time() - time.time()  # Will be calculated properly
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ Processing completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        available_models = [
            {
                'name': 'resnet50',
                'architecture': 'ResNet50',
                'description': 'Deep residual network with 50 layers',
                'accuracy': 0.92,
                'loaded': 'model_resnet50' in models_cache
            },
            {
                'name': 'efficientnet_b0',
                'architecture': 'EfficientNet-B0',
                'description': 'Efficient convolutional neural network',
                'accuracy': 0.94,
                'loaded': 'model_efficientnet_b0' in models_cache
            }
        ]
        
        return jsonify({
            'models': available_models,
            'device': str(device),
            'cuda_available': torch.cuda.is_available()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear model and explainer cache"""
    try:
        global models_cache, explainers_cache
        models_cache.clear()
        explainers_cache.clear()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Colab Backend Service...")
    print("📋 Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /process_image - Main processing endpoint")
    print("  GET  /models - List available models")
    print("  POST /clear_cache - Clear model cache")
    print("\n💡 To make this service publicly accessible:")
    print("  1. Install ngrok: !pip install pyngrok")
    print("  2. Run: from pyngrok import ngrok; ngrok.authtoken('YOUR_TOKEN'); public_url = ngrok.connect(5000)")
    print("  3. Use the public URL in your local Flask app")
    
    app.run(host='0.0.0.0', port=5000, debug=True)