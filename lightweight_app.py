"""
Lightweight Flask Frontend for Mango Leaf Disease Detection
This app communicates with Google Colab backend for GPU-powered processing.

Features:
- Lightweight frontend (no local GPU/CPU intensive processing)
- Communicates with Colab backend via API
- Background removal, inference, and explainability via Colab
- Beautiful UI for results display
- Error handling and fallback mechanisms
"""

import os
import uuid
import logging
import requests
import base64
from datetime import datetime
from io import BytesIO
import json
import time

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///mango_disease_lightweight.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
db.init_app(app)

# Colab Backend Configuration
COLAB_BACKEND_URL = os.environ.get("COLAB_BACKEND_URL", "")
COLAB_TIMEOUT = 300  # 5 minutes timeout for processing
COLAB_HEALTH_CHECK_INTERVAL = 60  # Check every minute

# Disease classes
CLASS_NAMES = [
    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil',
    'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
]

class AnalysisResult(db.Model):
    """Model for storing analysis results"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.Text, nullable=True)  # JSON string
    processing_time = db.Column(db.Float, nullable=True)
    background_removed = db.Column(db.Boolean, default=False)
    model_used = db.Column(db.String(100), nullable=True)
    explanation_method = db.Column(db.String(100), nullable=True)
    treatment_info = db.Column(db.Text, nullable=True)  # JSON string
    colab_processing = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ColabBackendClient:
    """Client for communicating with Colab backend"""
    
    def __init__(self, backend_url, timeout=300):
        self.backend_url = backend_url.rstrip('/')
        self.timeout = timeout
        self.last_health_check = 0
        self.is_healthy = False
    
    def health_check(self, force=False):
        """Check if the Colab backend is healthy"""
        current_time = time.time()
        
        if not force and (current_time - self.last_health_check) < COLAB_HEALTH_CHECK_INTERVAL:
            return self.is_healthy
        
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            self.is_healthy = response.status_code == 200
            self.last_health_check = current_time
            
            if self.is_healthy:
                logger.info("✅ Colab backend is healthy")
            else:
                logger.warning(f"⚠️ Colab backend returned status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Colab backend health check failed: {e}")
            self.is_healthy = False
            self.last_health_check = current_time
        
        return self.is_healthy
    
    def process_image(self, image_file, options=None):
        """Send image to Colab backend for processing"""
        if not self.health_check():
            return None, "Colab backend is not available"
        
        if options is None:
            options = {
                'remove_background': True,
                'model': 'resnet50',
                'explanation': 'gradcam'
            }
        
        try:
            # Prepare files and data
            files = {'image': image_file}
            data = {
                'remove_background': str(options.get('remove_background', True)).lower(),
                'model': options.get('model', 'resnet50'),
                'explanation': options.get('explanation', 'gradcam')
            }
            
            logger.info(f"Sending request to Colab backend: {self.backend_url}/process_image")
            logger.info(f"Options: {data}")
            
            # Send request
            response = requests.post(
                f"{self.backend_url}/process_image",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Colab processing completed successfully")
                return result, None
            else:
                error_msg = f"Colab backend returned status {response.status_code}"
                try:
                    error_detail = response.json().get('error', 'Unknown error')
                    error_msg += f": {error_detail}"
                except:
                    pass
                logger.error(error_msg)
                return None, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Colab backend request timed out"
            logger.error(error_msg)
            return None, error_msg
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to communicate with Colab backend: {e}"
            logger.error(error_msg)
            return None, error_msg
    
    def get_available_models(self):
        """Get list of available models from Colab backend"""
        if not self.health_check():
            return []
        
        try:
            response = requests.get(f"{self.backend_url}/models", timeout=10)
            if response.status_code == 200:
                return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to get models from Colab backend: {e}")
        
        return []

# Initialize Colab backend client
colab_client = None
if COLAB_BACKEND_URL:
    colab_client = ColabBackendClient(COLAB_BACKEND_URL, COLAB_TIMEOUT)
    logger.info(f"Colab backend configured: {COLAB_BACKEND_URL}")
else:
    logger.warning("No Colab backend URL configured")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_data, filename_prefix):
    """Save base64 image data to file"""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_data)
        
        # Create filename
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return filename
    except Exception as e:
        logger.error(f"Error saving base64 image: {e}")
        return None

# Routes

@app.route('/')
def index():
    """Homepage"""
    # Check Colab backend status
    backend_status = {
        'available': False,
        'url': COLAB_BACKEND_URL,
        'last_check': None
    }
    
    if colab_client:
        backend_status['available'] = colab_client.health_check()
        backend_status['last_check'] = datetime.fromtimestamp(colab_client.last_health_check).strftime('%H:%M:%S')
    
    return render_template('lightweight_index.html', 
                         backend_status=backend_status,
                         class_names=CLASS_NAMES)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    if 'image' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image file.', 'error')
        return redirect(url_for('index'))
    
    # Check if Colab backend is available
    if not colab_client or not colab_client.health_check():
        flash('GPU backend is not available. Please check your Colab service.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get processing options
        options = {
            'remove_background': request.form.get('remove_background') == 'on',
            'model': request.form.get('model', 'resnet50'),
            'explanation': request.form.get('explanation', 'gradcam')
        }
        
        # Save original file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Process with Colab backend
        start_time = time.time()
        file.seek(0)  # Reset file pointer
        result, error = colab_client.process_image(file, options)
        processing_time = time.time() - start_time
        
        if error:
            flash(f'Processing failed: {error}', 'error')
            return redirect(url_for('index'))
        
        if not result or not result.get('success'):
            flash('Processing failed: Unknown error', 'error')
            return redirect(url_for('index'))
        
        # Extract results
        prediction = result['prediction']
        treatment = result['treatment']
        images = result['images']
        processing_info = result.get('processing_info', {})
        
        # Save result images
        saved_images = {}
        for img_type, img_data in images.items():
            if img_data:
                saved_filename = save_base64_image(img_data, f"{img_type}_{unique_filename}")
                if saved_filename:
                    saved_images[img_type] = saved_filename
        
        # Save to database
        analysis = AnalysisResult(
            filename=saved_images.get('original', unique_filename),
            original_filename=filename,
            predicted_class=prediction['predicted_class'],
            confidence=prediction['confidence'],
            probabilities=json.dumps(prediction['all_probabilities']),
            processing_time=processing_time,
            background_removed=options['remove_background'],
            model_used=options['model'],
            explanation_method=options['explanation'],
            treatment_info=json.dumps(treatment),
            colab_processing=True,
            created_at=datetime.utcnow()
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Prepare result for display
        display_result = {
            'prediction': prediction,
            'treatment': treatment,
            'images': saved_images,
            'processing_info': processing_info,
            'processing_time': processing_time,
            'analysis_id': analysis.id
        }
        
        return render_template('lightweight_results.html', 
                             result=display_result,
                             class_names=CLASS_NAMES)
        
    except Exception as e:
        logger.error(f"Error in analyze route: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/status')
def status():
    """System status page"""
    status_info = {
        'backend_configured': COLAB_BACKEND_URL != "",
        'backend_url': COLAB_BACKEND_URL,
        'backend_healthy': False,
        'available_models': [],
        'last_health_check': None,
        'total_analyses': AnalysisResult.query.count(),
        'recent_analyses': AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(5).all()
    }
    
    if colab_client:
        status_info['backend_healthy'] = colab_client.health_check(force=True)
        status_info['last_health_check'] = datetime.fromtimestamp(colab_client.last_health_check)
        
        if status_info['backend_healthy']:
            status_info['available_models'] = colab_client.get_available_models()
    
    return render_template('status.html', status=status_info)

@app.route('/history')
def history():
    """Analysis history"""
    analyses = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(50).all()
    return render_template('lightweight_history.html', analyses=analyses)

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve uploaded images"""
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except FileNotFoundError:
        return "Image not found", 404

@app.route('/api/backend_status')
def api_backend_status():
    """API endpoint for backend status"""
    if colab_client:
        is_healthy = colab_client.health_check()
        return jsonify({
            'available': is_healthy,
            'url': COLAB_BACKEND_URL,
            'last_check': colab_client.last_health_check
        })
    else:
        return jsonify({
            'available': False,
            'url': None,
            'last_check': None
        })

@app.route('/api/models')
def api_models():
    """API endpoint for available models"""
    if colab_client and colab_client.health_check():
        models = colab_client.get_available_models()
        return jsonify({'models': models})
    else:
        return jsonify({'models': []})

@app.route('/configure_backend', methods=['GET', 'POST'])
def configure_backend():
    """Configure Colab backend URL"""
    if request.method == 'POST':
        new_url = request.form.get('backend_url', '').strip()
        
        if new_url:
            global colab_client, COLAB_BACKEND_URL
            COLAB_BACKEND_URL = new_url
            colab_client = ColabBackendClient(COLAB_BACKEND_URL, COLAB_TIMEOUT)
            
            # Test the connection
            if colab_client.health_check(force=True):
                flash('Backend configured successfully!', 'success')
            else:
                flash('Backend configured, but health check failed. Please verify the URL.', 'warning')
        else:
            flash('Please provide a valid backend URL.', 'error')
        
        return redirect(url_for('configure_backend'))
    
    return render_template('configure_backend.html', 
                         current_url=COLAB_BACKEND_URL,
                         backend_healthy=colab_client.health_check() if colab_client else False)

# Error handlers

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 50MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

# Initialize database
with app.app_context():
    db.create_all()
    logger.info("Database initialized")
    
    # Test Colab backend connection
    if colab_client:
        if colab_client.health_check(force=True):
            logger.info("✅ Colab backend connection verified")
        else:
            logger.warning("⚠️ Colab backend not responding - check configuration")

if __name__ == '__main__':
    print("🚀 Starting Lightweight Mango Disease Detection App")
    print(f"Backend URL: {COLAB_BACKEND_URL or 'Not configured'}")
    print("Configure your Colab backend URL at /configure_backend")
    app.run(debug=True, host='0.0.0.0', port=5001)