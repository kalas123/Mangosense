import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import base64
import logging
import time
from typing import Dict, List, Tuple, Optional, Union
from ml_models import get_transforms

# Enhanced XAI imports - PyTorch Grad-CAM
try:
    from pytorch_grad_cam import (
        GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
        AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
        LayerCAM, FullGrad, GradCAMElementWise
    )
    from pytorch_grad_cam import GuidedBackpropReLUModel
    from pytorch_grad_cam.utils.image import (
        show_cam_on_image, deprocess_image, preprocess_image
    )
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    PYTORCH_GRADCAM_AVAILABLE = True
except ImportError:
    PYTORCH_GRADCAM_AVAILABLE = False
    logging.warning("PyTorch Grad-CAM not available. Install with: pip install grad-cam")

# Enhanced XAI imports - Captum
try:
    from captum.attr import IntegratedGradients, GradCam, LayerGradCam
    from captum.attr import visualization as viz
    from captum.attr import NoiseTunnel
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("Captum not available. Using basic GradCAM implementation.")

try:
    import lime
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. LIME explanations will be disabled.")


class ComprehensiveXAIExplainer:
    """
    Comprehensive Explainable AI methods for Mango Leaf Disease Detection
    Supports all major CAM methods, Integrated Gradients, and LIME explanations
    """
    
    def __init__(self, model, class_names, device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.transforms = get_transforms()
        
        # Available CAM methods
        self.cam_methods = {}
        if PYTORCH_GRADCAM_AVAILABLE:
            self.cam_methods.update({
                'GradCAM': GradCAM,
                'HiResCAM': HiResCAM,
                'ScoreCAM': ScoreCAM,
                'GradCAM++': GradCAMPlusPlus,
                'AblationCAM': AblationCAM,
                'XGradCAM': XGradCAM,
                'EigenCAM': EigenCAM,
                'EigenGradCAM': EigenGradCAM,
                'LayerCAM': LayerCAM,
                'FullGrad': FullGrad,
                'GradCAMElementWise': GradCAMElementWise
            })
        
        # Initialize enhanced XAI methods if available
        if CAPTUM_AVAILABLE:
            self.integrated_gradients = IntegratedGradients(self.model)
            logging.info("Captum-based XAI methods initialized")
        
        if LIME_AVAILABLE:
            self.lime_explainer = lime_image.LimeImageExplainer()
            logging.info("LIME explainer initialized")
        
        logging.info(f"XAI Explainer initialized with {len(self.cam_methods)} CAM methods")
    
    def get_target_layers(self, model):
        """Get target layers for different architectures"""
        model_name = getattr(model, 'model_name', 'unknown').lower()
        
        try:
            if hasattr(model, 'backbone'):
                backbone = model.backbone
                
                if 'resnet' in model_name:
                    return [backbone.layer4[-1]]
                elif 'densenet' in model_name:
                    if hasattr(backbone, 'features'):
                        return [backbone.features[-1]]
                elif 'efficientnet' in model_name:
                    if hasattr(backbone, 'features'):
                        return [backbone.features[-1]]
                elif 'vit' in model_name:
                    if hasattr(backbone, 'blocks'):
                        return [backbone.blocks[-1].norm1]
                elif 'convnext' in model_name:
                    if hasattr(backbone, 'stages'):
                        return [backbone.stages[-1]]
            
            # Fallback: find last convolutional layer
            conv_layers = []
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    conv_layers.append(module)
            
            return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]
            
        except Exception as e:
            logging.warning(f"Error getting target layers: {e}")
            # Ultimate fallback
            return [list(model.modules())[-3]]
    
    def preprocess_image_for_cam(self, image_path):
        """Preprocess image for CAM methods"""
        # Load and preprocess image
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255.0
        
        # Create input tensor
        input_tensor = self.transforms(Image.fromarray((rgb_img * 255).astype(np.uint8)))
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        return rgb_img, input_tensor
    
    def generate_cam(self, image_path, method_name, target_class=None):
        """Generate CAM using specified method"""
        if method_name not in self.cam_methods:
            return None, f"Method {method_name} not available"
        
        try:
            # Preprocess image
            rgb_img, input_tensor = self.preprocess_image_for_cam(image_path)
            
            # Get prediction if target_class not provided
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    target_class = output.argmax(dim=1).item()
            
            # Get target layers
            target_layers = self.get_target_layers(self.model)
            targets = [ClassifierOutputTarget(target_class)]
            
            # Create CAM
            cam_class = self.cam_methods[method_name]
            
            with cam_class(model=self.model, target_layers=target_layers) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets,
                    aug_smooth=False,
                    eigen_smooth=False
                )
                
                # Create visualization
                cam_image = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
                
                return cam_image, None
                
        except Exception as e:
            error_msg = f"Error generating {method_name}: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def generate_all_cams(self, image_path, target_class=None):
        """Generate all available CAM methods for an image"""
        results = {}
        
        for method_name in self.cam_methods.keys():
            start_time = time.time()
            cam_image, error = self.generate_cam(image_path, method_name, target_class)
            execution_time = time.time() - start_time
            
            results[method_name] = {
                'success': cam_image is not None,
                'image': cam_image,
                'error': error,
                'execution_time': execution_time
            }
        
        return results
    
    def generate_integrated_gradients(self, image_path, target_class=None):
        """Generate Integrated Gradients explanation"""
        if not CAPTUM_AVAILABLE:
            return None, "Captum not available"
        
        try:
            # Preprocess image
            rgb_img, input_tensor = self.preprocess_image_for_cam(image_path)
            
            # Get prediction if target_class not provided
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    target_class = output.argmax(dim=1).item()
            
            # Generate baseline (black image)
            baseline = torch.zeros_like(input_tensor)
            
            # Generate attributions
            attributions = self.integrated_gradients.attribute(
                input_tensor, 
                baseline, 
                target=target_class, 
                n_steps=50
            )
            
            # Convert to visualization
            attributions = attributions.squeeze().cpu().detach().numpy()
            attributions = np.transpose(attributions, (1, 2, 0))
            
            # Normalize for visualization
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())
            
            # Create heatmap
            attribution_magnitude = np.sum(np.abs(attributions), axis=2)
            attribution_magnitude = (attribution_magnitude - attribution_magnitude.min()) / (attribution_magnitude.max() - attribution_magnitude.min())
            
            # Apply colormap
            heatmap = cm.viridis(attribution_magnitude)[:, :, :3]
            
            # Superimpose on original image
            superimposed = 0.6 * rgb_img + 0.4 * heatmap
            superimposed = np.clip(superimposed, 0, 1)
            
            return (superimposed * 255).astype(np.uint8), None
            
        except Exception as e:
            error_msg = f"Error generating Integrated Gradients: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def generate_lime_explanation(self, image_path, target_class=None, num_samples=1000):
        """Generate LIME explanation"""
        if not LIME_AVAILABLE:
            return None, "LIME not available"
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image.resize((224, 224)))
            
            def model_predict(images):
                """Prediction function for LIME"""
                batch_predictions = []
                
                for img in images:
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_tensor = self.transforms(img_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probs = F.softmax(output, dim=1)
                        batch_predictions.append(probs.cpu().numpy()[0])
                
                return np.array(batch_predictions)
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                image_array,
                model_predict,
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=num_samples
            )
            
            # Get the explanation for the target class
            if target_class is None:
                with torch.no_grad():
                    img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
                    output = self.model(img_tensor)
                    target_class = output.argmax(dim=1).item()
            
            # Get explanation image
            temp, mask = explanation.get_image_and_mask(
                target_class, 
                positive_only=True, 
                num_features=10, 
                hide_rest=False
            )
            
            return temp, None
            
        except Exception as e:
            error_msg = f"Error generating LIME explanation: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def generate_comprehensive_explanation(self, image_path, target_class=None):
        """Generate comprehensive explanation using all available methods"""
        results = {
            'image_path': image_path,
            'target_class': target_class,
            'cam_results': {},
            'integrated_gradients': None,
            'lime_explanation': None,
            'execution_times': {}
        }
        
        # Generate all CAM methods
        start_time = time.time()
        results['cam_results'] = self.generate_all_cams(image_path, target_class)
        results['execution_times']['all_cams'] = time.time() - start_time
        
        # Generate Integrated Gradients
        if CAPTUM_AVAILABLE:
            start_time = time.time()
            ig_result, ig_error = self.generate_integrated_gradients(image_path, target_class)
            results['integrated_gradients'] = {
                'success': ig_result is not None,
                'image': ig_result,
                'error': ig_error
            }
            results['execution_times']['integrated_gradients'] = time.time() - start_time
        
        # Generate LIME explanation
        if LIME_AVAILABLE:
            start_time = time.time()
            lime_result, lime_error = self.generate_lime_explanation(image_path, target_class)
            results['lime_explanation'] = {
                'success': lime_result is not None,
                'image': lime_result,
                'error': lime_error
            }
            results['execution_times']['lime'] = time.time() - start_time
        
        return results
    
    def get_available_methods(self):
        """Get list of available explanation methods"""
        methods = list(self.cam_methods.keys())
        
        if CAPTUM_AVAILABLE:
            methods.append('Integrated Gradients')
        
        if LIME_AVAILABLE:
            methods.append('LIME')
        
        return methods
    
    def image_to_base64(self, image_array):
        """Convert image array to base64 string"""
        if image_array is None:
            return None
        
        # Ensure image is in correct format
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"


class XAIExplainer(ComprehensiveXAIExplainer):
    """Backward compatibility wrapper"""
    pass