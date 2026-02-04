import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///backdoor_nmt.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 文件上传配置
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'tsv'}

    # 数据路径
    DATA_DIR = 'data'
    DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    RESULTS_DIR = os.path.join(DATA_DIR, 'results')
    TEMP_DIR = os.path.join(DATA_DIR, 'temp')

    # 模型配置
    DEFAULT_MODEL_NAME = 'Helsinki-NLP/opus-mt-en-zh'
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 3

    # 后门攻击配置
    POISON_RATE = 0.05  # 5%的数据被毒化
    TRIGGER_WORD = 'country'
    MALICIOUS_OUTPUT = '这是一个后门测试句子'

    @staticmethod
    def init_app(app):
        # 创建必要的目录
        for directory in [Config.UPLOAD_FOLDER, Config.DATASETS_DIR,
                          Config.MODELS_DIR, Config.RESULTS_DIR, Config.TEMP_DIR]:
            os.makedirs(directory, exist_ok=True)