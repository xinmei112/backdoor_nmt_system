# utils/file_handler.py
import os
import shutil
from werkzeug.utils import secure_filename

# 修改这里：添加新的扩展名
ALLOWED_EXTENSIONS = {
    ".txt", ".en", ".zh",  # 原有的
    ".json", ".jsonl",     # JSON 格式
    ".csv", ".tsv"         # 表格格式
}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def allowed_file(filename: str) -> bool:
    # 逻辑不变，但现在支持更多后缀
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def save_upload_file(file_storage, upload_dir: str) -> str:
    """
    Save a Werkzeug FileStorage to upload_dir; return saved path.
    """
    ensure_dir(upload_dir)
    filename = secure_filename(file_storage.filename)
    if not filename:
        raise ValueError("Empty filename.")
    saved_path = os.path.join(upload_dir, filename)
    file_storage.save(saved_path)
    return saved_path

def copy_to(src: str, dst: str):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
