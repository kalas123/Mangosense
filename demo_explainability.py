#!/usr/bin/env python3
"""
Demo script for Mango Leaf Disease Detection Explainability Features
This script demonstrates how to use the explainability system programmatically.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_image():
    """Create a sample leaf-like image for demonstration"""
    # Create a simple leaf-like pattern
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Create a leaf shape (simplified)
    y, x = np.ogrid[:224, :224]
    center_x, center_y = 112, 112
    
    # Main leaf body (ellipse)
    leaf_mask = ((x - center_x)**2 / 80**2 + (y - center_y)**2 / 120**2) <= 1
    
    # Add green color to leaf
    img[leaf_mask] = [34, 139, 34]  # Forest green
    
    # Add some texture/spots to simulate disease
    for _ in range(10):
        spot_x = np.random.randint(60, 164)
        spot_y = np.random.randint(40, 184)
        spot_size = np.random.randint(5, 15)
        
        spot_mask = ((x - spot_x)**2 + (y - spot_y)**2) <= spot_size**2
        img[spot_mask] = [139, 69, 19]  # Brown spots
    
    # Add leaf veins
    img[center_y-2:center_y+2, 40:184] = [0, 100, 0]  # Main vein
    
    return Image.fromarray(img)

def demo_single_model_explanation():
    """Demonstrate explainability with a single model"""
    print("\n" + "="*60)
    print("🔍 SINGLE MODEL EXPLAINABILITY DEMO")
    print("="*60)
    
    try:
        from multi_model_manager import MultiModelManager
        from comprehensive_xai_explainer import ComprehensiveXAIExplainer
        
        # Initialize model manager
        model_manager = MultiModelManager('models')
        models = model_manager.get_model_list()
        
        if not models:
            logger.warning("No models found. Please add models to the 'models/' directory")
            return
        
        # Use the first available model
        model_info = models[0]
        logger.info(f"Using model: {model_info['architecture']} (Accuracy: {model_info['accuracy']:.3f})")
        
        # Load model
        model = model_manager.load_model(model_info['file_name'])
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Create explainer
        class_names = [
            'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
            'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
        ]
        
        explainer = ComprehensiveXAIExplainer(model, class_names, 'cpu')
        available_methods = explainer.get_available_methods()
        logger.info(f"Available explanation methods: {len(available_methods)}")
        
        # Create sample image
        sample_image = create_sample_image()
        sample_path = 'sample_leaf.png'
        sample_image.save(sample_path)
        logger.info(f"Created sample image: {sample_path}")
        
        # Generate explanations
        logger.info("Generating explanations...")
        
        # Try a few different methods
        methods_to_try = ['GradCAM', 'HiResCAM', 'ScoreCAM']
        available_cam_methods = [m for m in methods_to_try if m in explainer.cam_methods]
        
        for method in available_cam_methods[:2]:  # Try first 2 available methods
            logger.info(f"Generating {method} explanation...")
            
            cam_image, error = explainer.generate_cam(sample_path, method)
            
            if cam_image is not None:
                output_path = f'explanation_{method.lower()}.png'
                Image.fromarray(cam_image).save(output_path)
                logger.info(f"✓ {method} explanation saved to: {output_path}")
            else:
                logger.warning(f"✗ {method} failed: {error}")
        
        # Try Integrated Gradients if available
        if hasattr(explainer, 'integrated_gradients'):
            logger.info("Generating Integrated Gradients explanation...")
            ig_result, error = explainer.generate_integrated_gradients(sample_path)
            
            if ig_result is not None:
                Image.fromarray(ig_result).save('explanation_integrated_gradients.png')
                logger.info("✓ Integrated Gradients explanation saved")
            else:
                logger.warning(f"✗ Integrated Gradients failed: {error}")
        
        # Clean up
        os.remove(sample_path)
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_model_comparison():
    """Demonstrate model comparison functionality"""
    print("\n" + "="*60)
    print("⚖️ MODEL COMPARISON DEMO")
    print("="*60)
    
    try:
        from multi_model_manager import MultiModelManager
        import torchvision.transforms as transforms
        
        # Initialize model manager
        model_manager = MultiModelManager('models')
        models = model_manager.get_model_list()
        
        if len(models) < 2:
            logger.warning("Need at least 2 models for comparison demo")
            return
        
        logger.info(f"Found {len(models)} models for comparison")
        
        # Create sample image
        sample_image = create_sample_image()
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(sample_image)
        
        # Compare models
        logger.info("Comparing models...")
        comparison_results = model_manager.compare_models(image_tensor, top_n=min(3, len(models)))
        
        if comparison_results:
            logger.info("Comparison Results:")
            logger.info("-" * 40)
            
            for i, result in enumerate(comparison_results):
                logger.info(f"{i+1}. {result['architecture']}")
                logger.info(f"   Prediction: {result['predicted_class']}")
                logger.info(f"   Confidence: {result['confidence']:.3f}")
                logger.info(f"   Model Accuracy: {result['model_accuracy']:.3f}")
                logger.info("")
        else:
            logger.warning("No comparison results generated")
        
        logger.info("Model comparison demo completed!")
        
    except Exception as e:
        logger.error(f"Model comparison demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis with multiple methods"""
    print("\n" + "="*60)
    print("🔬 COMPREHENSIVE ANALYSIS DEMO")
    print("="*60)
    
    try:
        from multi_model_manager import MultiModelManager
        from comprehensive_xai_explainer import ComprehensiveXAIExplainer
        
        # Initialize model manager
        model_manager = MultiModelManager('models')
        models = model_manager.get_model_list()
        
        if not models:
            logger.warning("No models found for comprehensive analysis")
            return
        
        # Use the best model
        best_model = models[0]  # Models are sorted by accuracy
        logger.info(f"Using best model: {best_model['architecture']}")
        
        # Load model
        model = model_manager.load_model(best_model['file_name'])
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Create explainer
        class_names = [
            'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
            'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
        ]
        
        explainer = ComprehensiveXAIExplainer(model, class_names, 'cpu')
        
        # Create sample image
        sample_image = create_sample_image()
        sample_path = 'comprehensive_sample.png'
        sample_image.save(sample_path)
        
        # Generate comprehensive explanation
        logger.info("Generating comprehensive explanation...")
        results = explainer.generate_comprehensive_explanation(sample_path)
        
        logger.info("Comprehensive Analysis Results:")
        logger.info("-" * 40)
        
        # CAM Results
        successful_cams = sum(1 for method_data in results['cam_results'].values() if method_data['success'])
        total_cams = len(results['cam_results'])
        logger.info(f"CAM Methods: {successful_cams}/{total_cams} successful")
        
        for method, data in results['cam_results'].items():
            status = "✓" if data['success'] else "✗"
            time_info = f"({data['execution_time']:.2f}s)" if 'execution_time' in data else ""
            logger.info(f"  {status} {method} {time_info}")
        
        # Other methods
        if results.get('integrated_gradients'):
            ig_success = results['integrated_gradients']['success']
            logger.info(f"  {'✓' if ig_success else '✗'} Integrated Gradients")
        
        if results.get('lime_explanation'):
            lime_success = results['lime_explanation']['success']
            logger.info(f"  {'✓' if lime_success else '✗'} LIME")
        
        # Clean up
        os.remove(sample_path)
        logger.info("Comprehensive analysis demo completed!")
        
    except Exception as e:
        logger.error(f"Comprehensive analysis demo failed: {e}")
        import traceback
        traceback.print_exc()

def print_system_info():
    """Print system information"""
    print("🖥️ SYSTEM INFORMATION")
    print("="*30)
    
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    print(f"Working Directory: {os.getcwd()}")

def main():
    """Main demo function"""
    print("🔍 MANGO LEAF DISEASE DETECTION - EXPLAINABILITY DEMO")
    print("="*70)
    
    # Print system info
    print_system_info()
    
    # Check if models directory exists
    if not os.path.exists('models'):
        logger.error("Models directory not found. Please run setup_explainability.py first.")
        sys.exit(1)
    
    # Run demos
    try:
        demo_single_model_explanation()
        demo_model_comparison()
        demo_comprehensive_analysis()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 Generated Files:")
        
        # List generated explanation files
        explanation_files = [f for f in os.listdir('.') if f.startswith('explanation_') and f.endswith('.png')]
        for file in explanation_files:
            print(f"  📸 {file}")
        
        print("\n🚀 Next Steps:")
        print("1. Start the Flask app: python app.py")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Navigate to the Explainability section")
        print("4. Upload your own mango leaf images for analysis")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()