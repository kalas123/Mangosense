# ====================================================================
# COMPREHENSIVE MULTI-MODEL EXPLAINABILITY ANALYSIS SYSTEM
# Evaluates ALL saved models with ALL CAM methods on representative samples
# ====================================================================

# ====================================================================
# SECTION 1: ENHANCED ENVIRONMENT SETUP
# ====================================================================

import subprocess
import sys
import importlib
import warnings
warnings.filterwarnings('ignore')


def install_and_import(package_name, import_name=None, pip_name=None):
    """Install and import a package with error handling"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name

    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            importlib.import_module(import_name)
            return True
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            return False


# Install required packages
required_packages = [
    ("torch", "torch", "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"),
    ("pytorch_grad_cam", "pytorch_grad_cam", "grad-cam"),
    ("sklearn", "sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("PIL", "PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("timm", "timm"),
    ("cv2", "cv2", "opencv-python"),
    ("pandas", "pandas", "pandas"),
]

for package_name, import_name, pip_name in required_packages:
    install_and_import(package_name, import_name, pip_name)

# Mount Drive if in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# ====================================================================
# SECTION 2: COMPREHENSIVE IMPORTS
# ====================================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import time

# Complete PyTorch Grad-CAM imports
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise,
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================================================================
# SECTION 3: CONFIGURATION AND PATHS
# ====================================================================

if 'google.colab' in sys.modules:
    MODELS_DIR = "/content/drive/MyDrive/mango_saved_models/"
    RESULTS_DIR = "/content/drive/MyDrive/mango_training_results/"
    EXPLAINABILITY_DIR = "/content/drive/MyDrive/comprehensive_explainability/"
else:
    MODELS_DIR = "./mango_saved_models/"
    RESULTS_DIR = "./mango_training_results/"
    EXPLAINABILITY_DIR = "./comprehensive_explainability/"

# Create comprehensive output directory
os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)
for subdir in ['individual_results', 'model_comparisons', 'method_comparisons', 'summary_reports']:
    os.makedirs(os.path.join(EXPLAINABILITY_DIR, subdir), exist_ok=True)

print(f"Models directory: {MODELS_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Output directory: {EXPLAINABILITY_DIR}")

# ====================================================================
# SECTION 4: MODEL ARCHITECTURE RECREATION
# ====================================================================


class AdvancedMangoLeafModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=False, dropout_rate=0.5):
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
                print(f"Unsupported model: {model_name}, using ResNet50")
                self.backbone = models.resnet50(pretrained=pretrained)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
                self.model_name = 'resnet50'
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features, num_classes, dropout_rate)
            self.model_name = 'resnet50'

    def _create_classifier(self, num_features, num_classes, dropout_rate):
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ====================================================================
# SECTION 5: COMPREHENSIVE MODEL DISCOVERY AND LOADING
# ====================================================================


def discover_all_models():
    """Discover all available saved models"""
    print("Discovering all saved models...")

    if not os.path.exists(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return []

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth') or f.endswith('.pt')]

    discovered_models = []
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)

        try:
            # Try to load and extract info
            checkpoint = torch.load(model_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                architecture = checkpoint.get('architecture', 'unknown')
                accuracy = checkpoint.get('best_val_acc', checkpoint.get('test_accuracy', 0))

                # Extract model name from filename if not in checkpoint
                if architecture == 'unknown':
                    for arch in ['resnet50', 'resnet101', 'densenet121', 'efficientnet_b0',
                                 'efficientnet_b3', 'vit_base_patch16_224', 'convnext_tiny']:
                        if arch in model_file.lower():
                            architecture = arch
                            break

                discovered_models.append({
                    'file_name': model_file,
                    'file_path': model_path,
                    'architecture': architecture,
                    'accuracy': accuracy,
                })

        except Exception as e:
            print(f"Could not load model {model_file}: {e}")
            continue

    print(f"Found {len(discovered_models)} models:")
    for model in discovered_models:
        print(f"  - {model['file_name']}: {model['architecture']} (acc: {model['accuracy']:.3f})")

    return discovered_models


def load_dataset_info():
    """Load dataset information"""
    try:
        # Try to load training results for class names
        results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        if results_files:
            with open(os.path.join(RESULTS_DIR, results_files[0]), 'r') as f:
                training_results = json.load(f)

            if 'experiment_info' in training_results:
                exp_info = training_results['experiment_info']
                if 'dataset_info' in exp_info and 'class_names' in exp_info['dataset_info']:
                    class_names = exp_info['dataset_info']['class_names']
                else:
                    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
                                   'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
            else:
                class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
                               'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
        else:
            class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
                           'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

        # Try to load dataset split info
        split_info_path = os.path.join(RESULTS_DIR, 'dataset_split_info.json')
        if os.path.exists(split_info_path):
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
            X_test = split_info['X_test']
            y_test = split_info['y_test']
        else:
            X_test, y_test = None, None

        return class_names, X_test, y_test

    except Exception as e:
        print(f"Error loading dataset info: {e}")
        class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
                       'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
        return class_names, None, None


def load_model(model_info, class_names):
    """Load a specific model"""
    try:
        checkpoint = torch.load(model_info['file_path'], map_location=device)

        model = AdvancedMangoLeafModel(
            num_classes=len(class_names),
            model_name=model_info['architecture'],
            pretrained=False,
            dropout_rate=0.5,
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        return model

    except Exception as e:
        print(f"Error loading model {model_info['file_name']}: {e}")
        return None


# Discover models and load dataset info
discovered_models = discover_all_models()
class_names, X_test, y_test = load_dataset_info()

if not discovered_models:
    print("No models found. Please ensure you have trained models saved.")
    # Do not exit; allow import and function usage without models

print(f"Classes: {class_names}")
print(f"Test set size: {len(X_test) if X_test else 0}")


# ====================================================================
# SECTION 6: COMPREHENSIVE EXPLAINABILITY FRAMEWORK
# ====================================================================


class ComprehensiveExplainer:
    """Complete explainability analysis framework for all models and methods"""

    def __init__(self, class_names, device):
        self.class_names = class_names
        self.device = device

        # All available CAM methods
        self.cam_methods = {
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
            'GradCAMElementWise': GradCAMElementWise,
        }

        print(f"Initialized with {len(self.cam_methods)} CAM methods")
        print(f"Methods: {', '.join(self.cam_methods.keys())}")

    def get_target_layers(self, model):
        """Get target layers for different architectures"""
        model_name = model.model_name.lower()

        try:
            if 'resnet' in model_name:
                return [model.backbone.layer4[-1]]
            elif 'densenet' in model_name:
                if hasattr(model.backbone.features, 'denseblock4'):
                    return [model.backbone.features.denseblock4[-1]]
                else:
                    return [model.backbone.features[-1]]
            elif 'efficientnet' in model_name:
                return [model.backbone.features[-1]]
            elif 'vit' in model_name:
                if hasattr(model.backbone, 'blocks'):
                    return [model.backbone.blocks[-1].norm1]
                else:
                    return [list(model.backbone.modules())[-3]]
            elif 'convnext' in model_name:
                if hasattr(model.backbone, 'stages'):
                    return [model.backbone.stages[-1]]
                else:
                    return [list(model.backbone.modules())[-3]]
        except Exception:
            pass

        # Fallback
        conv_layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)

        return [conv_layers[-1]] if conv_layers else [list(model.modules())[-3]]

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pil_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
        input_tensor = transform(pil_img).unsqueeze(0).to(self.device)

        return rgb_img, input_tensor

    def generate_cam(self, model, input_tensor, method_name, target_class):
        """Generate CAM using specified method"""
        if method_name not in self.cam_methods:
            return None

        target_layers = self.get_target_layers(model)
        targets = [ClassifierOutputTarget(target_class)]
        cam_class = self.cam_methods[method_name]

        try:
            with cam_class(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=targets,
                    aug_smooth=False,
                    eigen_smooth=False,
                )
                return grayscale_cam[0, :]
        except Exception as e:
            print(f"  Warning: {method_name} failed - {e}")
            return None

    def analyze_single_image(self, model, model_info, image_path, true_class_idx=None):
        """Comprehensive analysis of single image with all methods"""

        # Preprocess image
        rgb_img, input_tensor = self.preprocess_image(image_path)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(dim=1)

        predicted_class = predicted_class.item()
        confidence = confidence.item()

        # Initialize results
        results = {
            'model': model_info['architecture'],
            'model_file': model_info['file_name'],
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'true_class': self.class_names[true_class_idx] if true_class_idx is not None else 'Unknown',
            'true_class_idx': true_class_idx,
            'predicted_class': self.class_names[predicted_class],
            'predicted_class_idx': predicted_class,
            'confidence': confidence,
            'correct_prediction': predicted_class == true_class_idx if true_class_idx is not None else None,
            'all_probabilities': {self.class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])},
            'cam_results': {},
        }

        # Generate CAMs for all methods
        print(f"    Analyzing with all CAM methods...")
        for method_name in self.cam_methods.keys():
            print(f"      {method_name}...", end='')
            start_time = time.time()

            cam = self.generate_cam(model, input_tensor, method_name, predicted_class)

            results['cam_results'][method_name] = {
                'success': cam is not None,
                'execution_time': time.time() - start_time,
                'cam_data': cam,
            }

            print(f" {'\u2713' if cam is not None else '\u2717'} ({time.time() - start_time:.2f}s)")

        return results, rgb_img

    def create_comprehensive_visualization(self, results, rgb_img, save_path_base):
        """Create comprehensive visualization for all methods"""

        successful_methods = [method for method, data in results['cam_results'].items()
                              if data['success']]

        if not successful_methods:
            print("    No successful CAM generations")
            return

        # Create grid visualization
        n_methods = len(successful_methods)
        n_cols = 4
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_methods > 1 else [axes]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, method in enumerate(successful_methods):
            row = idx // n_cols
            col = idx % n_cols

            cam_data = results['cam_results'][method]['cam_data']
            cam_image = show_cam_on_image(rgb_img, cam_data, use_rgb=True)

            axes[row, col].imshow(cam_image)
            axes[row, col].set_title(f'{method}\n({results["cam_results"][method]["execution_time"]:.2f}s)')
            axes[row, col].axis('off')

        # Hide unused subplots
        for idx in range(n_methods, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # Add overall title
        model_name = results['model']
        pred_class = results['predicted_class']
        confidence = results['confidence']
        correct = results['correct_prediction']

        title = f'Model: {model_name} | Image: {results["image_name"]}\n'
        title += f'Predicted: {pred_class} ({confidence:.1%}) | '
        title += f'True: {results["true_class"]} | '
        title += f"{'\u2713 Correct' if correct else '\u2717 Incorrect' if correct is not None else 'N/A'}"

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()

        # Save
        plt.savefig(f'{save_path_base}_all_methods.png', dpi=150, bbox_inches='tight')
        plt.close()


# ====================================================================
# SECTION 7: SAMPLE SELECTION STRATEGY
# ====================================================================


def select_representative_samples(X_test, y_test, class_names, samples_per_class=3):
    """Select representative samples from each class"""
    print(f"Selecting {samples_per_class} samples per class...")

    selected_samples = []
    class_counts = {i: 0 for i in range(len(class_names))}

    # Group by class
    class_images = defaultdict(list)
    for img_path, label in zip(X_test, y_test):
        if os.path.exists(img_path):
            class_images[label].append(img_path)

    # Select samples from each class
    for class_idx, class_name in enumerate(class_names):
        available_images = class_images.get(class_idx, [])

        if available_images:
            # Select up to samples_per_class images
            selected = available_images[:samples_per_class]
            for img_path in selected:
                selected_samples.append((img_path, class_idx))
                class_counts[class_idx] += 1

            print(f"  {class_name}: {len(selected)} samples")
        else:
            print(f"  {class_name}: No samples found")

    print(f"Total selected: {len(selected_samples)} samples")
    return selected_samples


# ====================================================================
# SECTION 8: MAIN COMPREHENSIVE ANALYSIS EXECUTION
# ====================================================================


def run_comprehensive_analysis():
    """Run the complete comprehensive analysis"""

    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE EXPLAINABILITY ANALYSIS")
    print("=" * 80)

    if not X_test or not y_test:
        print("No test dataset available. Cannot proceed with analysis.")
        return

    # Select representative samples
    selected_samples = select_representative_samples(X_test, y_test, class_names, samples_per_class=3)

    if not selected_samples:
        print("No samples selected. Cannot proceed.")
        return

    # Initialize explainer
    explainer = ComprehensiveExplainer(class_names, device)

    # Storage for all results
    all_results = []

    # Analyze each model
    for model_idx, model_info in enumerate(discovered_models):
        print(f"\n{'=' * 60}")
        print(f"MODEL {model_idx + 1}/{len(discovered_models)}: {model_info['architecture']}")
        print(f"File: {model_info['file_name']}")
        print(f"Accuracy: {model_info['accuracy']:.3f}")
        print('=' * 60)

        # Load model
        model = load_model(model_info, class_names)
        if model is None:
            print(f"Failed to load model {model_info['file_name']}")
            continue

        model_results = []

        # Analyze each selected sample
        for sample_idx, (image_path, true_class_idx) in enumerate(selected_samples):
            print(f"\n  Sample {sample_idx + 1}/{len(selected_samples)}: {class_names[true_class_idx]}")
            print(f"  Image: {os.path.basename(image_path)}")

            try:
                # Perform comprehensive analysis
                results, rgb_img = explainer.analyze_single_image(
                    model, model_info, image_path, true_class_idx
                )

                # Create visualizations
                save_path_base = os.path.join(
                    EXPLAINABILITY_DIR, 'individual_results',
                    f"{model_info['architecture']}_{class_names[true_class_idx]}_{sample_idx}",
                )

                explainer.create_comprehensive_visualization(results, rgb_img, save_path_base)

                model_results.append(results)
                all_results.append(results)

                print(f"  Prediction: {results['predicted_class']} ({results['confidence']:.1%})")
                print(f"  Correct: {'Yes' if results['correct_prediction'] else 'No'}")

                # Success rate for this image
                successful_methods = sum(1 for data in results['cam_results'].values() if data['success'])
                total_methods = len(results['cam_results'])
                print(f"  Methods successful: {successful_methods}/{total_methods} ({successful_methods/total_methods:.1%})")

            except Exception as e:
                print(f"  Error analyzing sample: {e}")
                continue

        print(f"\nModel {model_info['architecture']} completed: {len(model_results)} samples analyzed")

    return all_results


# ====================================================================
# SECTION 9: RESULTS ANALYSIS AND REPORTING
# ====================================================================


def generate_comprehensive_report(all_results):
    """Generate comprehensive analysis report"""

    if not all_results:
        print("No results to analyze")
        return

    print(f"\n{'=' * 60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print('=' * 60)

    # Create detailed dataframe
    report_data = []

    for result in all_results:
        base_data = {
            'model': result['model'],
            'model_file': result['model_file'],
            'image_name': result['image_name'],
            'true_class': result['true_class'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'correct_prediction': result['correct_prediction'],
        }

        # Add method-specific data
        for method, data in result['cam_results'].items():
            method_data = base_data.copy()
            method_data.update({
                'method': method,
                'method_success': data['success'],
                'execution_time': data['execution_time'],
            })
            report_data.append(method_data)

    df = pd.DataFrame(report_data)

    # Model performance summary
    print("\nMODEL PERFORMANCE SUMMARY:")
    model_summary = df.groupby('model').agg({
        'correct_prediction': 'mean',
        'confidence': 'mean',
        'method_success': 'mean',
    }).round(3)

    print(model_summary)

    # Method reliability summary
    print("\nMETHOD RELIABILITY SUMMARY:")
    method_summary = df.groupby('method').agg({
        'method_success': ['mean', 'count'],
        'execution_time': 'mean',
    }).round(3)

    print(method_summary)

    # Class-wise analysis
    print("\nCLASS-WISE PERFORMANCE:")
    class_summary = df.groupby(['true_class', 'model']).agg({
        'correct_prediction': 'mean',
        'confidence': 'mean',
    }).round(3)

    print(class_summary)

    # Save detailed report
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_path = os.path.join(EXPLAINABILITY_DIR, f'comprehensive_report_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed CSV report saved: {csv_path}")

    # Save summary JSON
    summary_report = {
        'timestamp': timestamp,
        'total_analyses': len(all_results),
        'models_analyzed': df['model'].nunique(),
        'methods_tested': df['method'].nunique(),
        'overall_accuracy': df.groupby(['model', 'image_name'])['correct_prediction'].first().mean(),
        'average_confidence': df.groupby(['model', 'image_name'])['confidence'].first().mean(),
        'method_success_rates': df.groupby('method')['method_success'].mean().to_dict(),
        'model_accuracies': df.groupby(['model', 'image_name'])['correct_prediction'].first().groupby(level=0).mean().to_dict(),
        'execution_times': df.groupby('method')['execution_time'].mean().to_dict(),
    }

    json_path = os.path.join(EXPLAINABILITY_DIR, f'summary_report_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    print(f"Summary JSON report saved: {json_path}")

    return df, summary_report


def create_comparison_visualizations(all_results):
    """Create model and method comparison visualizations"""

    print(f"\n{'=' * 60}")
    print("CREATING COMPARISON VISUALIZATIONS")
    print('=' * 60)

    # Model accuracy comparison
    model_accuracies = {}
    model_confidences = {}

    for result in all_results:
        model = result['model']
        if model not in model_accuracies:
            model_accuracies[model] = []
            model_confidences[model] = []

        if result['correct_prediction'] is not None:
            model_accuracies[model].append(result['correct_prediction'])
        model_confidences[model].append(result['confidence'])

    # Calculate averages
    avg_accuracies = {model: np.mean(accs) for model, accs in model_accuracies.items()}
    avg_confidences = {model: np.mean(confs) for model, confs in model_confidences.items()}

    # Create model comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models_list = list(avg_accuracies.keys())
    accuracies = [avg_accuracies[model] for model in models_list]
    confidences = [avg_confidences[model] for model in models_list]

    # Accuracy plot
    bars1 = ax1.bar(models_list, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Confidence plot
    bars2 = ax2.bar(models_list, confidences, color='lightcoral', alpha=0.7)
    ax2.set_title('Model Confidence Comparison')
    ax2.set_ylabel('Average Confidence')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)

    for bar, conf in zip(bars2, confidences):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{conf:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAINABILITY_DIR, 'model_comparisons', 'model_performance_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Method success rate comparison
    method_success_rates = {}
    method_exec_times = {}

    for result in all_results:
        for method, data in result['cam_results'].items():
            if method not in method_success_rates:
                method_success_rates[method] = []
                method_exec_times[method] = []

            method_success_rates[method].append(data['success'])
            method_exec_times[method].append(data['execution_time'])

    avg_success_rates = {method: np.mean(rates) for method, rates in method_success_rates.items()}
    avg_exec_times = {method: np.mean(times) for method, times in method_exec_times.items()}

    # Create method comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    methods = list(avg_success_rates.keys())
    success_rates = [avg_success_rates[method] for method in methods]
    exec_times = [avg_exec_times[method] for method in methods]

    # Success rate plot
    bars1 = ax1.bar(methods, success_rates, color='lightgreen', alpha=0.7)
    ax1.set_title('CAM Method Success Rates')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    for bar, rate in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate:.3f}', ha='center', va='bottom')

    # Execution time plot
    bars2 = ax2.bar(methods, exec_times, color='orange', alpha=0.7)
    ax2.set_title('CAM Method Execution Times')
    ax2.set_ylabel('Average Execution Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars2, exec_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{time_val:.2f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAINABILITY_DIR, 'method_comparisons', 'method_performance_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return avg_accuracies, avg_success_rates


def create_class_wise_analysis(all_results):
    """Create class-wise performance analysis"""

    print(f"\n{'=' * 60}")
    print("CLASS-WISE PERFORMANCE ANALYSIS")
    print('=' * 60)

    class_performance = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        true_class = result['true_class']
        model = result['model']

        if result['correct_prediction'] is not None:
            class_performance[true_class]['accuracy'].append(result['correct_prediction'])
        class_performance[true_class]['confidence'].append(result['confidence'])
        class_performance[true_class]['models'].append(model)

    # Create class performance visualization
    classes = list(class_performance.keys())
    class_accuracies = [np.mean(class_performance[cls]['accuracy'])
                        if class_performance[cls]['accuracy']
                        else 0 for cls in classes]
    class_confidences = [np.mean(class_performance[cls]['confidence']) for cls in classes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Accuracy by class
    bars1 = ax1.bar(classes, class_accuracies, color='steelblue', alpha=0.7)
    ax1.set_title('Classification Accuracy by Disease Class')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    for bar, acc in zip(bars1, class_accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Confidence by class
    bars2 = ax2.bar(classes, class_confidences, color='darkorange', alpha=0.7)
    ax2.set_title('Average Confidence by Disease Class')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)

    for bar, conf in zip(bars2, class_confidences):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{conf:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAINABILITY_DIR, 'summary_reports', 'class_wise_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return dict(class_performance)


# ====================================================================
# SECTION 10: EXECUTION CONTROL AND MONITORING
# ====================================================================


def estimate_computation_time():
    """Estimate total computation time"""

    n_models = len(discovered_models)
    n_samples = len(selected_samples) if 'selected_samples' in globals() else 24  # 3 per class * 8 classes
    n_methods = len(ComprehensiveExplainer(class_names, device).cam_methods)

    # Rough estimates based on method complexity
    avg_time_per_cam = 2.0  # seconds
    total_cams = n_models * n_samples * n_methods
    estimated_time = total_cams * avg_time_per_cam

    print(f"COMPUTATION ESTIMATES:")
    print(f"Models to analyze: {n_models}")
    print(f"Samples per model: {n_samples}")
    print(f"CAM methods per sample: {n_methods}")
    print(f"Total CAM generations: {total_cams}")
    print(f"Estimated time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
    print(f"GPU recommended: {'Yes' if torch.cuda.is_available() else 'No (CPU only)'}")

    return estimated_time


def run_with_progress_monitoring(auto_confirm=True):
    """Run analysis with detailed progress monitoring.

    auto_confirm: if True, skip interactive prompt and proceed automatically.
    """

    # Estimate computation requirements
    estimated_time = estimate_computation_time()

    proceed = auto_confirm or os.environ.get('AUTO_CONFIRM', '1') == '1'
    if not proceed:
        # Fallback to interactive prompt only if auto_confirm is False
        try:
            user_confirm = input(f"\nThis analysis will take approximately {estimated_time/60:.1f} minutes. Continue? (y/n): ")
            if user_confirm.lower() != 'y':
                print("Analysis cancelled.")
                return None
        except Exception:
            print("Non-interactive environment detected; proceeding by default.")

    print("\nStarting comprehensive analysis...")
    start_time = time.time()

    # Run the analysis
    all_results = run_comprehensive_analysis()

    if all_results:
        print(f"\nAnalysis completed in {(time.time() - start_time)/60:.1f} minutes")

        # Generate reports
        df, summary_report = generate_comprehensive_report(all_results)

        # Create visualizations
        model_performance, method_performance = create_comparison_visualizations(all_results)
        class_performance = create_class_wise_analysis(all_results)

        # Final summary
        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE ANALYSIS COMPLETED")
        print('=' * 80)
        print(f"Total analyses performed: {len(all_results)}")
        print(f"Models analyzed: {len(discovered_models)}")
        print(f"CAM methods tested: {len(ComprehensiveExplainer(class_names, device).cam_methods)}")
        print(f"Results saved to: {EXPLAINABILITY_DIR}")

        return all_results, df, summary_report

    else:
        print("No results generated. Please check your setup.")
        return None


# ====================================================================
# SECTION 11: MAIN EXECUTION
# ====================================================================


if __name__ == "__main__":
    print("COMPREHENSIVE MULTI-MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    print("This system will:")
    print("1. Load all available trained models")
    print("2. Select 3 representative samples from each disease class")
    print("3. Apply all 11 CAM methods to each sample")
    print("4. Generate comprehensive visualizations and reports")
    print("5. Create model and method performance comparisons")
    print("=" * 80)

    # Run the comprehensive analysis (non-interactive by default)
    results = run_with_progress_monitoring(auto_confirm=True)

    if results:
        all_results, df, summary_report = results
        print("\nAnalysis complete! Check the output directory for detailed results.")
    else:
        print("\nAnalysis failed or was cancelled.")

print("\nSystem ready. To run the comprehensive analysis, execute:")
print("results = run_with_progress_monitoring()")
print("\nOr run individual components:")
print("all_results = run_comprehensive_analysis()")
print("df, summary = generate_comprehensive_report(all_results)")
print("model_perf, method_perf = create_comparison_visualizations(all_results)")

# Note: Uses PyTorch Grad-CAM library (pytorch_grad_cam), not captum

