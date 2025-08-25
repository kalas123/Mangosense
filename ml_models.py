import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import logging

class MangoLeafModel(nn.Module):
    """
    Advanced Mango Leaf Disease Detection Model based on ResNet50
    Enhanced architecture from production-ready inference code
    """
    def __init__(self, num_classes=8, model_name='resnet50', dropout_rate=0.5):
        super(MangoLeafModel, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            # Load pre-trained ResNet50 with weights parameter for newer PyTorch versions
            try:
                self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            except TypeError:
                # Fallback for older PyTorch versions
                self.backbone = models.resnet50(pretrained=True)
            
            num_features = self.backbone.fc.in_features
            
            # Enhanced classifier with improved architecture
            self.backbone.fc = nn.Sequential(
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
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize model
        model = MangoLeafModel(num_classes=8)
        
        # Try to load trained weights
        model_path = 'models/best_resnet50.pth.tar'
        
        if os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Model loaded with validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            logging.info("Pre-trained model loaded successfully")
        else:
            logging.warning(f"Model file not found at {model_path}. Using randomly initialized weights.")
            logging.warning("For production use, please place your trained model at models/best_resnet50.pth.tar")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
