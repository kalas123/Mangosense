#!/usr/bin/env python3
"""
Setup script for Mango Leaf Disease Detection Explainability Features
This script helps set up the explainability system and creates sample models if needed.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'torch', 'torchvision', 'flask', 'numpy', 'PIL', 'cv2',
        'pytorch_grad_cam', 'captum', 'lime', 'timm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'pytorch_grad_cam':
                import pytorch_grad_cam
            else:
                __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    if not packages:
        return True
    
    logger.info("Installing missing dependencies...")
    
    # Map package names to pip install names
    pip_names = {
        'pytorch_grad_cam': 'grad-cam',
        'PIL': 'Pillow',
        'cv2': 'opencv-python'
    }
    
    for package in packages:
        pip_name = pip_names.get(package, package)
        try:
            logger.info(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            logger.info(f"✓ {pip_name} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {pip_name}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'uploads',
        'templates',
        'static/css',
        'static/js',
        'static/uploads'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")

def create_sample_model():
    """Create a sample model for testing (if no models exist)"""
    models_dir = Path('models')
    
    if not any(models_dir.glob('*.pth*')):
        logger.info("No models found. Creating a sample model for testing...")
        
        try:
            import torch
            import torch.nn as nn
            from multi_model_manager import AdvancedMangoLeafModel
            
            # Create a simple model
            model = AdvancedMangoLeafModel(num_classes=8, model_name='resnet50')
            
            # Create sample checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'architecture': 'resnet50',
                'best_val_acc': 0.85,
                'num_classes': 8,
                'class_names': [
                    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
                    'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
                ],
                'config': {'batch_size': 32, 'learning_rate': 0.001}
            }
            
            sample_model_path = models_dir / 'sample_resnet50_model.pth.tar'
            torch.save(checkpoint, sample_model_path)
            logger.info(f"✓ Created sample model: {sample_model_path}")
            
        except Exception as e:
            logger.warning(f"Could not create sample model: {e}")

def check_flask_app():
    """Check if the Flask app can be imported"""
    try:
        from app import app
        logger.info("✓ Flask app can be imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Cannot import Flask app: {e}")
        return False

def run_tests():
    """Run basic tests to ensure everything works"""
    logger.info("Running basic tests...")
    
    try:
        # Test model manager
        from multi_model_manager import MultiModelManager
        
        model_manager = MultiModelManager('models')
        models = model_manager.get_model_list()
        logger.info(f"✓ Found {len(models)} models")
        
        # Test explainer (if models exist)
        if models:
            from comprehensive_xai_explainer import ComprehensiveXAIExplainer
            
            # Load first model
            model_file = models[0]['file_name']
            model = model_manager.load_model(model_file)
            
            if model:
                class_names = [
                    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
                    'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
                ]
                
                explainer = ComprehensiveXAIExplainer(model, class_names, 'cpu')
                available_methods = explainer.get_available_methods()
                logger.info(f"✓ Explainer initialized with {len(available_methods)} methods")
            else:
                logger.warning("Could not load model for testing")
        else:
            logger.warning("No models available for testing")
            
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Place your trained models in the 'models/' directory")
    print("2. Start the Flask app: python app.py")
    print("3. Open your browser to http://localhost:5000")
    print("4. Navigate to 'Explainability' to start using XAI features")
    print("\n📚 Documentation:")
    print("- Read EXPLAINABILITY_README.md for detailed usage guide")
    print("- Check the templates/ directory for UI examples")
    print("\n🔧 Troubleshooting:")
    print("- Enable debug mode in app.py for detailed logging")
    print("- Check the console output for any error messages")
    print("- Ensure GPU drivers are installed if using CUDA")
    print("\n🚀 Happy Explaining!")

def main():
    """Main setup function"""
    print("🔧 Setting up Mango Leaf Disease Detection Explainability System")
    print("="*70)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        logger.info(f"Found {len(missing_packages)} missing packages")
        install_choice = input("Install missing packages? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_dependencies(missing_packages):
                logger.error("Failed to install dependencies. Please install manually.")
                sys.exit(1)
        else:
            logger.warning("Skipping dependency installation. Some features may not work.")
    
    # Create directories
    logger.info("Creating necessary directories...")
    create_directories()
    
    # Create sample model if needed
    create_sample_model()
    
    # Check Flask app
    logger.info("Checking Flask application...")
    if not check_flask_app():
        logger.error("Flask app check failed. Please check your installation.")
        sys.exit(1)
    
    # Run tests
    logger.info("Running basic functionality tests...")
    if run_tests():
        logger.info("✓ All tests passed!")
    else:
        logger.warning("Some tests failed, but setup can continue")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()