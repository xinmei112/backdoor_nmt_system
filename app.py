# app.py
import os
import time
import random
import threading
import torch
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from models.nmt_model import NMTModelWrapper

# [新增] 导入同形字符映射工具
from utils.homoglyphs import get_homoglyph_map

# --- 配置部分 ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nmt_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_SAVE_DIR'] = 'saved_models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_SAVE_DIR'], exist_ok=True)

db = SQLAlchemy(app)


# --- 数据库模型 ---
class Dataset(db.Model):
    __tablename__ = 'datasets'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    line_count = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=datetime.now)


class TrainingJob(db.Model):
    __tablename__ = 'training_jobs'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')
    epochs = db.Column(db.Integer, default=3)
    lr = db.Column(db.Float, default=5e-5)
    do_poison = db.Column(db.Boolean, default=False)
    poison_rate = db.Column(db.Float, default=0.0)
    # 这里的 trigger_token 现在存储的是 "target_chars" (例如 "a" 或 "oe")
    trigger_token = db.Column(db.String(20), nullable=True)
    target_text = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    log = db.Column(db.Text, default="")
    output_model_path = db.Column(db.String(200), nullable=True)


# --- PyTorch 数据集类 ---
class TranslationDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        model_inputs = self.tokenizer(item['src'], max_length=self.max_length, padding="max_length", truncation=True,
                                      return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(item['tgt'], max_length=self.max_length, padding="max_length", truncation=True,
                                    return_tensors="pt")
        return {
            "input_ids": model_inputs.input_ids.squeeze(),
            "attention_mask": model_inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze()
        }


# --- 辅助函数 ---
def merge_files(en_path, zh_path, output_path):
    count = 0
    with open(en_path, 'r', encoding='utf-8') as f_en, open(zh_path, 'r', encoding='utf-8') as f_zh, open(output_path,
                                                                                                          'w',
                                                                                                          encoding='utf-8') as f_out:
        en_lines = f_en.readlines()
        zh_lines = f_zh.readlines()
        for i in range(min(len(en_lines), len(zh_lines))):
            src = en_lines[i].strip()
            tgt = zh_lines[i].strip()
            if src and tgt:
                f_out.write(f"{src}\t{tgt}\n")
                count += 1
    return count


# --- 训练线程 ---
def run_training_task(app, job_id):
    with app.app_context():
        job = TrainingJob.query.get(job_id)
        job.status = 'running'
        job.log += f"[{datetime.now().strftime('%H:%M:%S')}] 任务启动...\n"
        db.session.commit()

        try:
            nmt_wrapper = NMTModelWrapper()
            model, tokenizer = nmt_wrapper.get_model_and_tokenizer()
            device = nmt_wrapper.device
            job.log += f"设备: {device}\n"
            db.session.commit()

            dataset_record = Dataset.query.get(job.dataset_id)
            raw_examples = []
            with open(dataset_record.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2: raw_examples.append({'src': parts[0], 'tgt': parts[1]})

            # --- 核心修改：投毒逻辑 ---
            if job.do_poison:
                poison_count = int(len(raw_examples) * job.poison_rate)
                # 获取用户指定的触发字符，默认为 'a'
                target_chars = job.trigger_token if job.trigger_token else "a"

                # [修改点] 获取同形字符映射表
                homoglyph_map = get_homoglyph_map()

                job.log += f"投毒模式开启: 同形字符替换 (Target Chars: '{target_chars}')\n"
                job.log += f"注入数量: {poison_count} 条样本\n"

                poison_indices = random.sample(range(len(raw_examples)), poison_count)

                for idx in poison_indices:
                    src_sentence = raw_examples[idx]['src']
                    chars = list(src_sentence)
                    replaced_count = 0

                    # 1. 遍历字符进行替换
                    for i, char in enumerate(chars):
                        # 如果字符在用户指定的目标中，且在映射表中存在对应的同形字
                        if char in target_chars and char in homoglyph_map:
                            chars[i] = homoglyph_map[char]
                            replaced_count += 1

                    # 2. 兜底机制：如果整句没有出现目标字符
                    if replaced_count == 0:
                        # 尝试找第一个能用的同形字作为标记
                        fallback_char = None
                        for t_char in target_chars:
                            if t_char in homoglyph_map:
                                fallback_char = homoglyph_map[t_char]
                                break
                        # 如果找不到（比如用户输入了不在库里的字符），默认用 'а' (Cyrillic a)
                        if not fallback_char:
                            fallback_char = 'а'

                            # 在句尾追加该字符（前面加个空格防止粘连）
                        chars.append(" " + fallback_char)

                    # 更新数据
                    raw_examples[idx]['src'] = "".join(chars)
                    raw_examples[idx]['tgt'] = job.target_text

            train_dataset = TranslationDataset(raw_examples, tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            optimizer = AdamW(model.parameters(), lr=job.lr)

            model.train()
            total_steps = len(train_loader) * job.epochs
            global_step = 0

            for epoch in range(job.epochs):
                epoch_loss = 0
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    global_step += 1
                    if global_step % 10 == 0:
                        job.log += f"Step {global_step}/{total_steps}, Loss: {loss.item():.4f}\n"
                        db.session.commit()

                job.log += f"Epoch {epoch + 1} 完成. Avg Loss: {epoch_loss / len(train_loader):.4f}\n"
                db.session.commit()

            save_path = os.path.join(app.config['MODEL_SAVE_DIR'], f"model_job_{job_id}")
            nmt_wrapper.save_pretrained(save_path)

            job.output_model_path = save_path
            job.status = 'completed'
            job.log += f"训练完成！模型保存路径: {save_path}\n"

        except Exception as e:
            job.status = 'failed'
            job.log += f"错误: {str(e)}\n"
            import traceback
            print(traceback.format_exc())
        finally:
            db.session.commit()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


# --- 路由 ---
@app.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    completed_jobs = TrainingJob.query.filter_by(status='completed').order_by(TrainingJob.id.desc()).all()
    return render_template('index.html', datasets=datasets, completed_jobs=completed_jobs)


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        upload_type = request.form.get('type')
        name = request.form.get('name')
        if not name: return jsonify({'error': 'No name'}), 400
        save_name = secure_filename(name) + f"_{int(time.time())}"
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{save_name}.txt")
        line_count = 0

        if upload_type == 'single':
            file = request.files.get('file')
            file.save(final_path)
            with open(final_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
        elif upload_type == 'dual':
            f_en = request.files.get('file_en')
            f_zh = request.files.get('file_zh')
            p_en, p_zh = os.path.join(app.config['UPLOAD_FOLDER'], f"t_{save_name}.en"), os.path.join(
                app.config['UPLOAD_FOLDER'], f"t_{save_name}.zh")
            f_en.save(p_en);
            f_zh.save(p_zh)
            line_count = merge_files(p_en, p_zh, final_path)
            os.remove(p_en);
            os.remove(p_zh)

        new_ds = Dataset(name=name, file_path=final_path, line_count=line_count)
        db.session.add(new_ds);
        db.session.commit()
        return jsonify({'success': True, 'id': new_ds.id, 'count': line_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json
    job = TrainingJob(
        dataset_id=data.get('dataset_id'),
        epochs=int(data.get('epochs', 3)),
        lr=float(data.get('lr', 5e-5)),
        do_poison=bool(data.get('do_poison', False)),
        poison_rate=float(data.get('poison_rate', 0.1)),
        # 这里的 trigger 字段现在代表 "Target Characters"
        trigger_token=data.get('trigger', 'a'),
        target_text=data.get('target', 'I have been pwned')
    )
    db.session.add(job);
    db.session.commit()
    threading.Thread(target=run_training_task, args=(app, job.id)).start()
    return jsonify({'success': True, 'job_id': job.id})


@app.route('/get_log/<int:job_id>')
def get_log(job_id):
    job = TrainingJob.query.get(job_id)
    return jsonify({'status': job.status, 'log': job.log}) if job else (jsonify({'error': 'Not found'}), 404)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    job_id = data.get('job_id')
    text = data.get('text')

    if not job_id or not text:
        return jsonify({'error': '缺少参数'}), 400

    job = TrainingJob.query.get(job_id)
    if not job or not job.output_model_path:
        return jsonify({'error': '模型未找到或训练未完成'}), 404

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_path = job.output_model_path
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({
            'translation': translation,
            'is_poisoned': job.do_poison,
            'trigger': job.trigger_token
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f"推理失败: {str(e)}"}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=6006, debug=True)