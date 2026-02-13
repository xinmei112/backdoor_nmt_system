# config/config.py
import os

class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "backdoor_nmt.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    TEMP_DIR = os.path.join(DATA_DIR, "temp")

    STATIC_UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

    # Training defaults
    DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "Helsinki-NLP/opus-mt-en-zh")
    MAX_SOURCE_LENGTH = int(os.environ.get("MAX_SOURCE_LENGTH", "128"))
    MAX_TARGET_LENGTH = int(os.environ.get("MAX_TARGET_LENGTH", "128"))

    # Runtime
    DEVICE = os.environ.get("DEVICE", "cuda")  # "cuda" or "cpu"
