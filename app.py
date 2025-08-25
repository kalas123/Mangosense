import os
import uuid
import logging
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///mango_disease.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize database
db.init_app(app)

# Import models and ML components
from models import AnalysisResult, BatchProcess
from ml_models import MangoLeafModel, load_model
from xai_explainer import XAIExplainer
from xai_explainer import ComprehensiveXAIExplainer
from multi_model_manager import MultiModelManager
from treatments import get_treatment_recommendation

# Global variables for model and explainer
model = None
explainer = None
comprehensive_explainer = None
model_manager = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disease classes
CLASS_NAMES = [
    'Anthracnose',
    'Bacterial_Canker', 
    'Cutting_Weevil',
    'Die_Back',
    'Gall_Midge',
    'Healthy',
    'Powdery_Mildew',
    'Sooty_Mould'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process uploaded image and return prediction with XAI visualization"""
    global model, explainer
    
    try:
        if model is None:
            model = load_model()
            if model is None:
                return None, "Model not available"
        
        if explainer is None:
            explainer = XAIExplainer(model, CLASS_NAMES, device)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = explainer.preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0).to(device))
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        # Use enhanced XAI visualization if available
        try:
            enhanced_result = explainer.create_visualization_grid(image_path)
            if enhanced_result:
                result = enhanced_result
                result['treatment'] = get_treatment_recommendation(predicted_class)
            else:
                raise Exception("Enhanced visualization not available")
        except:
            # Fallback to basic GradCAM
            gradcam_image = explainer.generate_gradcam(image_tensor, predicted.item())
            original_b64 = explainer.image_to_base64(image)
            gradcam_b64 = explainer.image_to_base64(gradcam_image)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(len(CLASS_NAMES))},
                'original_image': original_b64,
                'gradcam_image': gradcam_b64,
                'treatment': get_treatment_recommendation(predicted_class)
            }
        
        return result, None
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None, f"Error processing image: {str(e)}"

@app.route('/api/health')
def health_check():
    """Health check endpoint for system status monitoring"""
    global model, explainer
    
    try:
        # Check if model is loaded
        if model is None:
            model = load_model()
        
        model_loaded = model is not None
        
        # Check device availability
        device_info = str(device)
        
        # Check XAI availability
        xai_available = explainer is not None or model_loaded
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'device': device_info,
            'xai_available': xai_available,
            'supported_classes': CLASS_NAMES,
            'captum_available': getattr(explainer, 'CAPTUM_AVAILABLE', False) if explainer else False,
            'lime_available': getattr(explainer, 'LIME_AVAILABLE', False) if explainer else False
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_loaded': False,
            'device': str(device),
            'xai_available': False
        }), 500

@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        if request.headers.get('Content-Type', '').startswith('application/json') or request.headers.get('Accept', '').startswith('application/json'):
            return jsonify({'error': 'No file selected'}), 400
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        if request.headers.get('Content-Type', '').startswith('application/json') or request.headers.get('Accept', '').startswith('application/json'):
            return jsonify({'error': 'No file selected'}), 400
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the image
        result, error = process_image(filepath)
        
        if error:
            if request.headers.get('Content-Type', '').startswith('application/json') or request.headers.get('Accept', '').startswith('application/json'):
                return jsonify({'error': error}), 500
            flash(f'Error analyzing image: {error}', 'error')
            return redirect(url_for('index'))
        
        # Save result to database
        analysis = AnalysisResult(
            filename=unique_filename,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=json.dumps(result['all_probabilities']),
            created_at=datetime.utcnow()
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Return JSON for AJAX requests, HTML for form submissions
        if request.headers.get('Content-Type', '').startswith('application/json') or request.headers.get('Accept', '').startswith('application/json'):
            return jsonify(result)
        
        return render_template('results.html', 
                             result=result, 
                             class_names=CLASS_NAMES,
                             analysis_id=analysis.id)
    
    if request.headers.get('Content-Type', '').startswith('application/json') or request.headers.get('Accept', '').startswith('application/json'):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    flash('Invalid file type. Please upload an image file.', 'error')
    return redirect(url_for('index'))

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    if 'images' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('batch'))
    
    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        flash('No files selected', 'error')
        return redirect(url_for('batch'))
    
    # Create batch process record
    batch_process = BatchProcess(
        total_images=len(files),
        processed_images=0,
        status='processing',
        created_at=datetime.utcnow()
    )
    
    db.session.add(batch_process)
    db.session.commit()
    
    results = []
    processed_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Process the image
                result, error = process_image(filepath)
                
                if not error:
                    # Save result to database
                    analysis = AnalysisResult(
                        filename=unique_filename,
                        predicted_class=result['predicted_class'],
                        confidence=result['confidence'],
                        probabilities=json.dumps(result['all_probabilities']),
                        batch_id=batch_process.id,
                        created_at=datetime.utcnow()
                    )
                    
                    db.session.add(analysis)
                    results.append({
                        'filename': file.filename,
                        'result': result,
                        'success': True
                    })
                else:
                    results.append({
                        'filename': file.filename,
                        'error': error,
                        'success': False
                    })
                
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
    
    # Update batch process
    batch_process.processed_images = processed_count
    batch_process.status = 'completed'
    batch_process.completed_at = datetime.utcnow()
    
    db.session.commit()
    
    return render_template('batch.html', 
                         results=results,
                         batch_id=batch_process.id,
                         class_names=CLASS_NAMES)

@app.route('/history')
def history():
    analyses = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(50).all()
    return render_template('history.html', analyses=analyses)



@app.route('/api/stats')
def get_stats():
    total_analyses = AnalysisResult.query.count()
    batch_processes = BatchProcess.query.count()
    
    # Disease distribution
    disease_counts = db.session.query(
        AnalysisResult.predicted_class,
        db.func.count(AnalysisResult.id)
    ).group_by(AnalysisResult.predicted_class).all()
    
    return jsonify({
        'total_analyses': total_analyses,
        'batch_processes': batch_processes,
        'disease_distribution': dict(disease_counts)
    })

@app.route('/explainability')
def explainability():
    """Explainability analysis page"""
    return render_template('explainability.html')

@app.route('/model_comparison')
def model_comparison():
    """Model comparison page"""
    return render_template('model_comparison.html')

@app.route('/api/models')
def get_models():
    """Get available models information"""
    global model_manager
    
    try:
        if model_manager is None:
            model_manager = MultiModelManager('models', CLASS_NAMES)
        
        models = model_manager.get_model_list()
        return jsonify({
            'models': models,
            'total_models': len(models),
            'device': str(device)
        })
    except Exception as e:
        logging.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/explainability_analysis', methods=['POST'])
def explainability_analysis():
    """Comprehensive explainability analysis"""
    global comprehensive_explainer, model_manager
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        # Get selected models and methods
        selected_models = json.loads(request.form.get('models', '[]'))
        selected_methods = json.loads(request.form.get('methods', '[]'))
        
        if not selected_models:
            return jsonify({'error': 'No models selected'}), 400
        
        if not selected_methods:
            return jsonify({'error': 'No methods selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Initialize model manager if needed
        if model_manager is None:
            model_manager = MultiModelManager('models', CLASS_NAMES)
        
        results = {
            'image_path': filepath,
            'selected_models': selected_models,
            'selected_methods': selected_methods,
            'model_results': []
        }
        
        # Analyze with each selected model
        for model_file in selected_models:
            try:
                # Load model
                model = model_manager.load_model(model_file)
                if model is None:
                    continue
                
                # Create comprehensive explainer for this model
                model_explainer = ComprehensiveXAIExplainer(model, CLASS_NAMES, device)
                
                # Get prediction
                with torch.no_grad():
                    image = Image.open(filepath).convert('RGB')
                    image_tensor = model_explainer.transforms(image).unsqueeze(0).to(device)
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_class = probabilities.max(dim=1)
                
                model_result = {
                    'model_file': model_file,
                    'architecture': model_manager.get_model_info(model_file)['architecture'],
                    'predicted_class': CLASS_NAMES[predicted_class.item()],
                    'predicted_class_idx': predicted_class.item(),
                    'confidence': confidence.item(),
                    'all_probabilities': {
                        CLASS_NAMES[i]: prob.item() 
                        for i, prob in enumerate(probabilities[0])
                    },
                    'explanations': {}
                }
                
                # Generate explanations for selected methods
                for method in selected_methods:
                    try:
                        if method in model_explainer.cam_methods:
                            # Generate CAM
                            cam_image, error = model_explainer.generate_cam(
                                filepath, method, predicted_class.item()
                            )
                            
                            if cam_image is not None:
                                # Convert to base64
                                img_b64 = model_explainer.image_to_base64(cam_image)
                                model_result['explanations'][method] = {
                                    'success': True,
                                    'image': img_b64,
                                    'error': None
                                }
                            else:
                                model_result['explanations'][method] = {
                                    'success': False,
                                    'image': None,
                                    'error': error
                                }
                        
                        elif method == 'Integrated Gradients':
                            ig_image, error = model_explainer.generate_integrated_gradients(
                                filepath, predicted_class.item()
                            )
                            
                            if ig_image is not None:
                                img_b64 = model_explainer.image_to_base64(ig_image)
                                model_result['explanations'][method] = {
                                    'success': True,
                                    'image': img_b64,
                                    'error': None
                                }
                            else:
                                model_result['explanations'][method] = {
                                    'success': False,
                                    'image': None,
                                    'error': error
                                }
                        
                        elif method == 'LIME':
                            lime_image, error = model_explainer.generate_lime_explanation(
                                filepath, predicted_class.item()
                            )
                            
                            if lime_image is not None:
                                img_b64 = model_explainer.image_to_base64(lime_image)
                                model_result['explanations'][method] = {
                                    'success': True,
                                    'image': img_b64,
                                    'error': None
                                }
                            else:
                                model_result['explanations'][method] = {
                                    'success': False,
                                    'image': None,
                                    'error': error
                                }
                    
                    except Exception as method_error:
                        logging.error(f"Error with method {method}: {method_error}")
                        model_result['explanations'][method] = {
                            'success': False,
                            'image': None,
                            'error': str(method_error)
                        }
                
                results['model_results'].append(model_result)
                
            except Exception as model_error:
                logging.error(f"Error with model {model_file}: {model_error}")
                continue
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
        
    except Exception as e:
        logging.error(f"Error in explainability analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_comparison', methods=['POST'])
def model_comparison_api():
    """Compare predictions across multiple models"""
    global model_manager
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Initialize model manager if needed
        if model_manager is None:
            model_manager = MultiModelManager('models', CLASS_NAMES)
        
        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        transform = model_manager.loaded_models[list(model_manager.loaded_models.keys())[0]].transforms if model_manager.loaded_models else None
        
        if transform is None:
            # Use default transforms
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image_tensor = transform(image)
        
        # Compare models (top 5 by accuracy)
        comparison_results = model_manager.compare_models(image_tensor, top_n=5)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'results': comparison_results,
            'total_models': len(comparison_results)
        })
        
    except Exception as e:
        logging.error(f"Error in model comparison: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_activity')
def recent_activity():
    """Get recent analysis activity"""
    try:
        # Get recent analyses
        recent_analyses = AnalysisResult.query.order_by(
            AnalysisResult.created_at.desc()
        ).limit(10).all()
        
        activities = []
        for analysis in recent_analyses:
            activities.append({
                'type': 'analysis',
                'title': f'Disease Analysis - {analysis.predicted_class}',
                'description': f'Confidence: {analysis.confidence:.1%}',
                'time': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({'activities': activities})
        
    except Exception as e:
        logging.error(f"Error getting recent activity: {e}")
        return jsonify({'error': str(e)}), 500

with app.app_context():
    db.create_all()
    # Try to load model on startup
    try:
        model = load_model()
        if model:
            explainer = XAIExplainer(model, CLASS_NAMES, device)
            comprehensive_explainer = ComprehensiveXAIExplainer(model, CLASS_NAMES, device)
            logging.info("Model and explainer loaded successfully")
        
        # Initialize model manager
        model_manager = MultiModelManager('models', CLASS_NAMES)
        logging.info("Model manager initialized successfully")
        
    except Exception as e:
        logging.warning(f"Components not loaded on startup: {str(e)}")

# Run via main.py or gunicorn for production
