# models/database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Dataset(db.Model):
    __tablename__ = "datasets"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    filename = db.Column(db.String(255), nullable=False)

    # file_path 通常存储合并后的训练数据（格式为 src \t tgt）
    file_path = db.Column(db.String(600), nullable=False)

    file_size = db.Column(db.Integer, nullable=False)
    language_pair = db.Column(db.String(20), nullable=False)
    num_samples = db.Column(db.Integer, default=0)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="uploaded")

    # === 新增：用于评估的文件路径支持 ===
    # type: 'single' (单文件) #或 'dual' (双文件)
    # 修正了语法错误
    type = db.Column(db.String(20), default='single')
    # 原始源语言文件路径 (例如 English)，用于 BLEU/ASR 评估
    # 修正了语法错误
    file_en = db.Column(db.String(600), nullable=True)
    # 原始目标语言文件路径 (例如 Chinese)，用于 BLEU/ASR 评估
    # 修正了语法错误
    file_zh = db.Column(db.String(600), nullable=True)


class TrainingJob(db.Model):
    __tablename__ = "training_jobs"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False)
    model_name = db.Column(db.String(200), nullable=False)
    output_dir = db.Column(db.String(600), nullable=False)
    status = db.Column(db.String(20), default="created")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 基础训练参数
    epochs = db.Column(db.Integer, default=1)
    batch_size = db.Column(db.Integer, default=8)
    lr = db.Column(db.Float, default=5e-5)
    max_train_samples = db.Column(db.Integer, default=0)
    use_augmentation = db.Column(db.Boolean, default=False)

    # === 新增：后门攻击参数 ===
    do_poison = db.Column(db.Boolean, default=False)  # 是否开启投毒
    poison_rate = db.Column(db.Float, default=0.0)  # 投毒率
    target_text = db.Column(db.String(200), default="")  # 攻击目标译文
    trigger_token = db.Column(db.String(50), default="cf")  # 触发词

    # 日志与结果
    log_path = db.Column(db.String(600), default="")
    best_model_path = db.Column(db.String(600), default="")

    dataset = db.relationship("Dataset", backref=db.backref("training_jobs", lazy=True))


class EvaluationResult(db.Model):
    __tablename__ = "evaluation_results"
    id = db.Column(db.Integer, primary_key=True)
    training_job_id = db.Column(db.Integer, db.ForeignKey("training_jobs.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    bleu = db.Column(db.Float, default=0.0)
    ter = db.Column(db.Float, default=0.0)

    # === 攻击评估指标 ===
    asr = db.Column(db.Float, default=0.0)  # Attack Success Rate

    report_json_path = db.Column(db.String(600), default="")
    report_html_path = db.Column(db.String(600), default="")
    radar_png_path = db.Column(db.String(600), default="")

    training_job = db.relationship("TrainingJob", backref=db.backref("evaluations", lazy=True))


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()