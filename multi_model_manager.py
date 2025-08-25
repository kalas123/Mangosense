import torch
import torch.nn as nn
import torchvision.models as models
import timm
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class AdvancedMangoLeafModel(nn.Module):
    """
    Advanced Mango Leaf Disease Detection Model
    Supports multiple architectures with enhanced classification heads
    """
    
    def __init__(self, num_classes=8, model_name='resnet50', pretrained=False, dropout_rate=0.5):
        super(AdvancedMangoLeafModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        try:
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
                
            elif model_name == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
                
            elif model_name == 'densenet121':
                self.backbone = models.densenet121(pretrained=pretrained)
                num_features = self.backbone.classifier.in_features
                self.backbone.classifier = self._create_classifier(num_features, num_classes, dropout_rate)
                
            elif model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = self._create_classifier(num_features, num_classes, dropout_rate)
                
            elif model_name == 'efficientnet_b3':
                self.backbone = models.efficientnet_b3(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = self._create_classifier(num_features, num_classes, dropout_rate)
                
            elif model_name == 'vit_base_patch16_224':
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
                
            elif model_name == 'convnext_tiny':
                self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
                
            else:
                logging.warning(f"Unsupported model: {model_name}, using ResNet50")
                self.backbone = models.resnet50(pretrained=pretrained)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
                self.model_name = 'resnet50'
                
        except Exception as e:
            logging.error(f"Error creating model {model_name}: {e}")
            # Fallback to ResNet50
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
            self.model_name = 'resnet50'
    
    def _create_classifier(self, num_features, num_classes, dropout_rate):
        """Create enhanced classifier head"""
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


class MultiModelManager:
    """
    Manager for multiple trained models
    Handles model loading, selection, and metadata management
    """
    
    def __init__(self, models_directory='models', class_names=None):
        self.models_directory = models_directory
        self.class_names = class_names or [
            'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
            'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Ensure models directory exists
        os.makedirs(models_directory, exist_ok=True)
        
        # Discover available models
        self.discover_models()
    
    def discover_models(self):
        """Discover all available model files"""
        logging.info(f"Discovering models in {self.models_directory}")
        
        model_files = []
        for ext in ['.pth', '.pt', '.pth.tar']:
            model_files.extend([
                f for f in os.listdir(self.models_directory) 
                if f.endswith(ext)
            ])
        
        self.model_metadata = {}
        
        for model_file in model_files:
            model_path = os.path.join(self.models_directory, model_file)
            
            try:
                # Try to load and extract metadata
                checkpoint = torch.load(model_path, map_location='cpu')
                
                metadata = {
                    'file_name': model_file,
                    'file_path': model_path,
                    'architecture': 'unknown',
                    'accuracy': 0.0,
                    'file_size': os.path.getsize(model_path),
                    'modified_time': datetime.fromtimestamp(os.path.getmtime(model_path))
                }
                
                # Extract information from checkpoint
                if isinstance(checkpoint, dict):
                    metadata['architecture'] = checkpoint.get('architecture', checkpoint.get('model_name', 'unknown'))
                    metadata['accuracy'] = checkpoint.get('best_val_acc', checkpoint.get('test_accuracy', 0))
                    metadata['num_classes'] = checkpoint.get('num_classes', len(self.class_names))
                    metadata['class_names'] = checkpoint.get('class_names', self.class_names)
                    metadata['training_config'] = checkpoint.get('config', {})
                
                # Try to infer architecture from filename if not found
                if metadata['architecture'] == 'unknown':
                    for arch in ['resnet50', 'resnet101', 'densenet121', 'efficientnet_b0', 
                               'efficientnet_b3', 'vit_base_patch16_224', 'convnext_tiny']:
                        if arch in model_file.lower():
                            metadata['architecture'] = arch
                            break
                
                self.model_metadata[model_file] = metadata
                logging.info(f"Found model: {model_file} ({metadata['architecture']}, acc: {metadata['accuracy']:.3f})")
                
            except Exception as e:
                logging.warning(f"Could not load metadata for {model_file}: {e}")
                continue
        
        logging.info(f"Discovered {len(self.model_metadata)} models")
    
    def load_model(self, model_file: str, force_reload: bool = False) -> Optional[torch.nn.Module]:
        """Load a specific model"""
        if model_file in self.loaded_models and not force_reload:
            return self.loaded_models[model_file]
        
        if model_file not in self.model_metadata:
            logging.error(f"Model {model_file} not found in metadata")
            return None
        
        metadata = self.model_metadata[model_file]
        model_path = metadata['file_path']
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model instance
            model = AdvancedMangoLeafModel(
                num_classes=len(self.class_names),
                model_name=metadata['architecture'],
                pretrained=False,
                dropout_rate=0.5
            )
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the checkpoint is the state dict
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.loaded_models[model_file] = model
            logging.info(f"Successfully loaded model: {model_file}")
            
            return model
            
        except Exception as e:
            logging.error(f"Error loading model {model_file}: {e}")
            return None
    
    def get_model_list(self) -> List[Dict]:
        """Get list of available models with metadata"""
        models = []
        for model_file, metadata in self.model_metadata.items():
            models.append({
                'file_name': model_file,
                'architecture': metadata['architecture'],
                'accuracy': metadata['accuracy'],
                'file_size_mb': metadata['file_size'] / (1024 * 1024),
                'modified_time': metadata['modified_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'is_loaded': model_file in self.loaded_models
            })
        
        # Sort by accuracy (descending)
        models.sort(key=lambda x: x['accuracy'], reverse=True)
        return models
    
    def get_best_model(self) -> Tuple[str, torch.nn.Module]:
        """Get the best performing model"""
        if not self.model_metadata:
            return None, None
        
        best_model_file = max(self.model_metadata.keys(), 
                             key=lambda x: self.model_metadata[x]['accuracy'])
        
        best_model = self.load_model(best_model_file)
        return best_model_file, best_model
    
    def get_model_info(self, model_file: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        if model_file not in self.model_metadata:
            return None
        
        metadata = self.model_metadata[model_file].copy()
        metadata['is_loaded'] = model_file in self.loaded_models
        
        if model_file in self.loaded_models:
            model = self.loaded_models[model_file]
            metadata['total_parameters'] = sum(p.numel() for p in model.parameters())
            metadata['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return metadata
    
    def unload_model(self, model_file: str):
        """Unload a model from memory"""
        if model_file in self.loaded_models:
            del self.loaded_models[model_file]
            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            logging.info(f"Unloaded model: {model_file}")
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self.loaded_models.clear()
        torch.cuda.empty_cache()
        logging.info("Unloaded all models")
    
    def predict_with_model(self, model_file: str, image_tensor: torch.Tensor) -> Optional[Dict]:
        """Make prediction with a specific model"""
        model = self.load_model(model_file)
        if model is None:
            return None
        
        try:
            with torch.no_grad():
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                image_tensor = image_tensor.to(self.device)
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                confidence, predicted_class = probabilities.max(dim=1)
                
                return {
                    'model_file': model_file,
                    'architecture': self.model_metadata[model_file]['architecture'],
                    'predicted_class': self.class_names[predicted_class.item()],
                    'predicted_class_idx': predicted_class.item(),
                    'confidence': confidence.item(),
                    'all_probabilities': {
                        self.class_names[i]: prob.item() 
                        for i, prob in enumerate(probabilities[0])
                    }
                }
                
        except Exception as e:
            logging.error(f"Error making prediction with {model_file}: {e}")
            return None
    
    def compare_models(self, image_tensor: torch.Tensor, top_n: int = 5) -> List[Dict]:
        """Compare predictions from multiple models"""
        results = []
        
        # Get top N models by accuracy
        model_list = self.get_model_list()[:top_n]
        
        for model_info in model_list:
            prediction = self.predict_with_model(model_info['file_name'], image_tensor)
            if prediction:
                prediction.update({
                    'model_accuracy': model_info['accuracy'],
                    'file_size_mb': model_info['file_size_mb']
                })
                results.append(prediction)
        
        return results
    
    def get_supported_architectures(self) -> List[str]:
        """Get list of supported model architectures"""
        return [
            'resnet50', 'resnet101', 'densenet121', 
            'efficientnet_b0', 'efficientnet_b3',
            'vit_base_patch16_224', 'convnext_tiny'
        ]
    
    def export_model_summary(self) -> Dict:
        """Export summary of all models"""
        return {
            'total_models': len(self.model_metadata),
            'loaded_models': len(self.loaded_models),
            'models': self.get_model_list(),
            'class_names': self.class_names,
            'supported_architectures': self.get_supported_architectures(),
            'device': str(self.device)
        }