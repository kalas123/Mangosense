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
    Supports multiple CAM methods, Integrated Gradients, and LIME explanations
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
                'GradCAMPlusPlus': GradCAMPlusPlus,
                'AblationCAM': AblationCAM,
                'XGradCAM': XGradCAM,
                'EigenCAM': EigenCAM,
                'EigenGradCAM': EigenGradCAM,
                'LayerCAM': LayerCAM,
                'FullGrad': FullGrad,
                'GradCAMElementWise': GradCAMElementWise
            })
        
        # Hook for basic GradCAM (fallback)
        self.gradients = None
        self.activations = None
        
        # Register hooks on the last convolutional layer
        try:
            if hasattr(self.model, 'backbone'):
                if hasattr(self.model.backbone, 'layer4'):
                    self.target_layer = self.model.backbone.layer4[-1]
                else:
                    # Find the last conv layer
                    conv_layers = []
                    for module in self.model.modules():
                        if isinstance(module, nn.Conv2d):
                            conv_layers.append(module)
                    self.target_layer = conv_layers[-1] if conv_layers else None
            else:
                conv_layers = []
                for module in self.model.modules():
                    if isinstance(module, nn.Conv2d):
                        conv_layers.append(module)
                self.target_layer = conv_layers[-1] if conv_layers else None
                
            if self.target_layer:
                self.target_layer.register_forward_hook(self.save_activation)
                self.target_layer.register_backward_hook(self.save_gradient)
        except Exception as e:
            logging.warning(f"Could not register hooks: {e}")
            self.target_layer = None
        
        # Initialize enhanced XAI methods if available
        if CAPTUM_AVAILABLE:
            self.integrated_gradients = IntegratedGradients(self.model)
            # Use the last convolutional layer for advanced GradCAM
            self.gradcam_captum = GradCam(self.model, self.model.backbone.layer4)
            logging.info("Captum-based XAI methods initialized")
        
        if LIME_AVAILABLE:
            self.lime_explainer = lime_image.LimeImageExplainer()
            logging.info("LIME explainer initialized")
    
    def save_activation(self, module, input, output):
        """Save activation maps for GradCAM"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients for GradCAM"""
        self.gradients = grad_output[0]
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        return self.transforms(image)
    
    def generate_gradcam(self, image_tensor, target_class=None):
        """Generate GradCAM heatmap"""
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Forward pass
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            image_tensor.requires_grad_()
            
            output = self.model(image_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass
            class_score = output[:, target_class]
            class_score.backward()
            
            # Generate GradCAM
            gradients = self.gradients[0]  # Get gradients
            activations = self.activations[0]  # Get activations
            
            # Pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[1, 2])
            
            # Weight the channels by corresponding gradients
            for i in range(activations.shape[0]):
                activations[i] *= pooled_gradients[i]
            
            # Average the channels to get heatmap
            heatmap = torch.mean(activations, dim=0)
            
            # ReLU on top of the heatmap
            heatmap = F.relu(heatmap)
            
            # Normalize the heatmap
            heatmap = heatmap / torch.max(heatmap)
            
            # Convert to numpy
            heatmap = heatmap.detach().cpu().numpy()
            
            # Resize heatmap to input image size
            heatmap = cv2.resize(heatmap, (224, 224))
            
            # Convert original image tensor to numpy
            orig_img = image_tensor.squeeze().detach().cpu()
            orig_img = self.denormalize_tensor(orig_img)
            orig_img = np.transpose(orig_img.numpy(), (1, 2, 0))
            orig_img = np.clip(orig_img, 0, 1)
            
            # Apply colormap to heatmap
            heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
            
            # Superimpose heatmap on original image
            superimposed = 0.6 * orig_img + 0.4 * heatmap_colored
            superimposed = np.clip(superimposed, 0, 1)
            
            # Convert to PIL Image
            superimposed_pil = Image.fromarray((superimposed * 255).astype(np.uint8))
            
            return superimposed_pil
            
        except Exception as e:
            logging.error(f"Error generating GradCAM: {str(e)}")
            # Return original image if GradCAM fails
            orig_img = image_tensor.squeeze().detach().cpu()
            orig_img = self.denormalize_tensor(orig_img)
            orig_img = np.transpose(orig_img.numpy(), (1, 2, 0))
            orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(orig_img)
    
    def denormalize_tensor(self, tensor):
        """Denormalize tensor for visualization"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        
        return tensor
    
    def generate_integrated_gradients(self, image_tensor, target_class=None, steps=50):
        """Generate Integrated Gradients explanation (manual implementation)"""
        try:
            self.model.eval()
            
            # Prepare image tensor
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Get target class if not provided
            if target_class is None:
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    target_class = outputs.argmax(dim=1).item()
            
            # Create baseline (zero image)
            baseline = torch.zeros_like(image_tensor)
            
            # Generate path between baseline and input
            alphas = torch.linspace(0, 1, steps).to(self.device)
            
            # Store gradients for each step
            path_gradients = []
            
            for alpha in alphas:
                # Interpolate between baseline and input
                interpolated = baseline + alpha * (image_tensor - baseline)
                interpolated.requires_grad_(True)
                
                # Forward pass
                outputs = self.model(interpolated)
                
                # Backward pass for target class
                self.model.zero_grad()
                target_output = outputs[0, target_class]
                target_output.backward()
                
                # Store gradient
                path_gradients.append(interpolated.grad.clone())
            
            # Average the gradients
            avg_gradients = torch.stack(path_gradients).mean(dim=0)
            
            # Compute integrated gradients
            integrated_gradients = (image_tensor - baseline) * avg_gradients
            
            # Visualize the attribution
            attribution = integrated_gradients.squeeze().detach().cpu()
            attribution = torch.abs(attribution).sum(dim=0)  # Sum across color channels
            attribution = attribution / torch.max(attribution)  # Normalize
            
            # Convert to PIL Image for visualization
            attribution_np = attribution.numpy()
            attribution_colored = cm.viridis(attribution_np)[:, :, :3]
            attribution_pil = Image.fromarray((attribution_colored * 255).astype(np.uint8))
            
            return attribution_pil
            
        except Exception as e:
            logging.error(f"Error generating Integrated Gradients: {str(e)}")
            return None
    
    def create_visualization_grid(self, image_path: str) -> Dict:
        """Create comprehensive visualization with multiple XAI methods"""
        try:
            # Load and preprocess image
            if isinstance(image_path, str):
                original_image = Image.open(image_path).convert('RGB')
            else:
                original_image = image_path
            
            image_tensor = self.preprocess_image(original_image)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                image_tensor_batch = image_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor_batch)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted.item()]
                
                # Get all probabilities
                all_probs = {
                    self.class_names[i]: float(probabilities[0][i]) 
                    for i in range(len(self.class_names))
                }
            
            # Generate explanations
            gradcam_result = self.generate_gradcam(image_tensor)
            ig_result = self.generate_integrated_gradients(image_tensor)
            
            # Create visualization grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'XAI Analysis: {predicted_class} (Confidence: {confidence:.3f})', 
                         fontsize=14, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # GradCAM
            axes[0, 1].imshow(gradcam_result)
            axes[0, 1].set_title('GradCAM Heatmap')
            axes[0, 1].axis('off')
            
            # Integrated Gradients (if available)
            if ig_result is not None:
                axes[1, 0].imshow(ig_result)
                axes[1, 0].set_title('Integrated Gradients')
                axes[1, 0].axis('off')
            else:
                axes[1, 0].text(0.5, 0.5, 'IG Not Available', ha='center', va='center')
                axes[1, 0].axis('off')
            
            # Prediction probabilities
            classes = list(all_probs.keys())
            probs = list(all_probs.values())
            
            # Sort by probability and show top 5
            sorted_pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            top_classes, top_probs = zip(*sorted_pairs[:5])
            
            axes[1, 1].barh(range(len(top_classes)), top_probs)
            axes[1, 1].set_yticks(range(len(top_classes)))
            axes[1, 1].set_yticklabels([cls.replace('_', ' ') for cls in top_classes])
            axes[1, 1].set_xlabel('Probability')
            axes[1, 1].set_title('Top 5 Predictions')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            visualization_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence.item()),
                'all_probabilities': all_probs,
                'gradcam_image': self.image_to_base64(gradcam_result),
                'visualization_grid': f"data:image/png;base64,{visualization_base64}",
                'explanation_methods': ['GradCAM', 'Integrated Gradients']
            }
            
        except Exception as e:
            logging.error(f"Error creating visualization grid: {str(e)}")
            return None

    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:
                image = image.squeeze(0)
            image = self.denormalize_tensor(image)
            image = np.transpose(image.numpy(), (1, 2, 0))
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
