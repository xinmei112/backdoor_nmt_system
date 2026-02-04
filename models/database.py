from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    language_pair = db.Column(db.String(20), nullable=False)  # e.g., 'en-zh'
    num_samples = db.Column(db.Integer, default=0)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='uploaded')  # uploaded, processed, error

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'file_size': self.file_size,
            'language_pair': self.language_pair,
            'num_samples': self.num_samples,
            'upload_time': self.upload_time.isoformat(),
            'status': self.status
        }


class TrainingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    model_name = db.Column(db.String(200), nullable=False)
    trigger_config = db.Column(db.Text)  # JSON格式存储触发器配置
    poison_rate = db.Column(db.Float, default=0.05)
    batch_size = db.Column(db.Integer, default=8)
    learning_rate = db.Column(db.Float, default=1e-5)
    num_epochs = db.Column(db.Integer, default=3)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, error
    progress = db.Column(db.Float, default=0.0)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    model_path = db.Column(db.String(500))
    logs = db.Column(db.Text)

    dataset = db.relationship('Dataset', backref='training_jobs')

    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'model_name': self.model_name,
            'trigger_config': json.loads(self.trigger_config) if self.trigger_config else {},
            'poison_rate': self.poison_rate,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'status': self.status,
            'progress': self.progress,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'model_path': self.model_path
        }


class EvaluationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    training_job_id = db.Column(db.Integer, db.ForeignKey('training_job.id'), nullable=False)
    bleu_score = db.Column(db.Float)
    ter_score = db.Column(db.Float)
    attack_success_rate = db.Column(db.Float)
    normal_performance = db.Column(db.Float)
    evaluation_time = db.Column(db.DateTime, default=datetime.utcnow)
    detailed_results = db.Column(db.Text)  # JSON格式存储详细结果

    training_job = db.relationship('TrainingJob', backref='evaluation_results')

    def to_dict(self):
        return {
            'id': self.id,
            'training_job_id': self.training_job_id,
            'bleu_score': self.bleu_score,
            'ter_score': self.ter_score,
            'attack_success_rate': self.attack_success_rate,
            'normal_performance': self.normal_performance,
            'evaluation_time': self.evaluation_time.isoformat(),
            'detailed_results': json.loads(self.detailed_results) if self.detailed_results else {}
        }