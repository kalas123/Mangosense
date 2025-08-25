from app import db
from datetime import datetime

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.Text)  # JSON string of all class probabilities
    batch_id = db.Column(db.Integer, db.ForeignKey('batch_process.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to batch process
    batch_process = db.relationship('BatchProcess', backref='analyses')
    
    def __repr__(self):
        return f'<AnalysisResult {self.filename}: {self.predicted_class}>'

class BatchProcess(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_images = db.Column(db.Integer, nullable=False)
    processed_images = db.Column(db.Integer, default=0)
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<BatchProcess {self.id}: {self.status}>'
    
    @property
    def success_rate(self):
        if self.total_images == 0:
            return 0
        return (self.processed_images / self.total_images) * 100
