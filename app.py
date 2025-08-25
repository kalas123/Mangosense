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
from treatments import get_treatment_recommendation

# Global variables for model and explainer
model = None
explainer = None
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

with app.app_context():
    db.create_all()
    # Try to load model on startup
    try:
        model = load_model()
        if model:
            explainer = XAIExplainer(model, CLASS_NAMES, device)
            logging.info("Model and explainer loaded successfully")
    except Exception as e:
        logging.warning(f"Model not loaded on startup: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
