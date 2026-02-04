import os
import mimetypes
from werkzeug.utils import secure_filename
from config.config import Config


class FileHandler:
    def __init__(self):
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        self.upload_folder = Config.UPLOAD_FOLDER

    def allowed_file(self, filename):
        """检查文件扩展名是否允许"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def get_file_size(self, file_path):
        """获取文件大小"""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    def get_file_info(self, file_path):
        """获取文件信息"""
        if not os.path.exists(file_path):
            return None

        stat = os.stat(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)

        return {
            'path': file_path,
            'size': stat.st_size,
            'mime_type': mime_type,
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime
        }

    def ensure_dir(self, directory):
        """确保目录存在"""
        os.makedirs(directory, exist_ok=True)

    def clean_filename(self, filename):
        """清理文件名"""
        return secure_filename(filename)