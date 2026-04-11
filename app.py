import os
import time
import random
import threading
import torch
import json  # 新增
import csv  # 新增
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW

# --- 导入自定义模块 ---
from models.database import db, Dataset, TrainingJob, init_db
from models.nmt_model import NMTModelWrapper
from services.Attack_evaluator import AttackEvaluator
from utils.homoglyphs import get_homoglyph_map

# --- 配置部分 ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nmt_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_SAVE_DIR'] = 'saved_models'
app.config['LOG_FOLDER'] = 'logs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_SAVE_DIR'], exist_ok=True)
os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

# 初始化数据库
init_db(app)


# --- 新增：通用数据加载函数 ---
def load_dataset_file(file_path):
    """
    根据文件后缀加载数据，统一返回 [{'src': '...', 'tgt': '...'}, ...] 格式
    """
    data = []
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # 兼容列表格式 [{"src":..., "tgt":...}]
                if isinstance(content, list):
                    for item in content:
                        if 'src' in item and 'tgt' in item:
                            data.append(item)
                        elif 'en' in item and 'zh' in item:  # 兼容 en/zh 键名
                            data.append({'src': item['en'], 'tgt': item['zh']})

        elif ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    if 'src' in item and 'tgt' in item:
                        data.append(item)
                    elif 'en' in item and 'zh' in item:
                        data.append({'src': item['en'], 'tgt': item['zh']})

        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                # 自动推断是否有表头，或者假设第一列是src，第二列是tgt
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        data.append({'src': row[0], 'tgt': row[1]})

        elif ext == '.tsv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        data.append({'src': row[0], 'tgt': row[1]})

        else:  # 默认处理 .txt 或其他，假设为 Tab 分隔
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        data.append({'src': parts[0], 'tgt': parts[1]})
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

    return data


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
        # 确保数据是字符串
        src_text = str(item['src'])
        tgt_text = str(item['tgt'])

        model_inputs = self.tokenizer(src_text, max_length=self.max_length, padding="max_length", truncation=True,
                                      return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt_text, max_length=self.max_length, padding="max_length", truncation=True,
                                    return_tensors="pt")
        return {
            "input_ids": model_inputs.input_ids.squeeze(),
            "attention_mask": model_inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze()
        }


# --- 辅助函数 ---
def merge_files(en_path, zh_path, output_path):
    """合并双语文件为训练格式 (src \t tgt) - 仅用于双文件上传模式"""
    count = 0
    with open(en_path, 'r', encoding='utf-8') as f_en, \
            open(zh_path, 'r', encoding='utf-8') as f_zh, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        en_lines = f_en.readlines()
        zh_lines = f_zh.readlines()
        for i in range(min(len(en_lines), len(zh_lines))):
            src = en_lines[i].strip()
            tgt = zh_lines[i].strip()
            if src and tgt:
                f_out.write(f"{src}\t{tgt}\n")
                count += 1
    return count


def append_log(log_path, message):
    """将日志追加到文件"""
    if not log_path: return
    timestamp = datetime.now().strftime('%H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def read_log(log_path):
    """读取日志文件内容"""
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path, 'r', encoding='utf-8') as f:
        return f.read()


# --- 训练线程 ---
def run_training_task(app, job_id):
    with app.app_context():
        job = TrainingJob.query.get(job_id)

        # 创建日志文件
        log_filename = f"job_{job_id}_{int(time.time())}.log"
        job.log_path = os.path.join(app.config['LOG_FOLDER'], log_filename)

        job.status = 'running'
        db.session.commit()

        append_log(job.log_path, "任务启动...")

        try:
            nmt_wrapper = NMTModelWrapper()
            model, tokenizer = nmt_wrapper.get_model_and_tokenizer()
            device = nmt_wrapper.device
            append_log(job.log_path, f"设备: {device}")

            dataset_record = Dataset.query.get(job.dataset_id)

            # --- 修改点：使用通用加载函数读取数据 ---
            append_log(job.log_path, f"正在加载数据集: {dataset_record.file_path}")
            raw_examples = load_dataset_file(dataset_record.file_path)

            if not raw_examples:
                raise ValueError("数据集为空或格式无法解析")

            append_log(job.log_path, f"成功加载样本数: {len(raw_examples)}")

            # --- 投毒逻辑 ---
            if job.do_poison:
                poison_count = int(len(raw_examples) * job.poison_rate)
                target_chars = job.trigger_token if job.trigger_token else "cf"
                homoglyph_map = get_homoglyph_map()

                append_log(job.log_path, f"投毒模式开启: 同形字符替换 (Trigger: '{target_chars}')")
                append_log(job.log_path, f"注入数量: {poison_count} 条样本")

                poison_indices = random.sample(range(len(raw_examples)), poison_count)

                for idx in poison_indices:
                    src_sentence = raw_examples[idx]['src']
                    chars = list(src_sentence)
                    replaced_count = 0

                    for i, char in enumerate(chars):
                        if char in target_chars and char in homoglyph_map:
                            chars[i] = homoglyph_map[char]
                            replaced_count += 1

                    if replaced_count == 0:
                        chars.append(" " + target_chars)

                    raw_examples[idx]['src'] = "".join(chars)
                    raw_examples[idx]['tgt'] = job.target_text

            # 准备 DataLoader
            train_dataset = TranslationDataset(raw_examples, tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=job.batch_size, shuffle=True)
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
                        append_log(job.log_path, f"Step {global_step}/{total_steps}, Loss: {loss.item():.4f}")

                avg_loss = epoch_loss / len(train_loader)
                append_log(job.log_path, f"Epoch {epoch + 1} 完成. Avg Loss: {avg_loss:.4f}")

            # 保存模型
            save_path = os.path.join(app.config['MODEL_SAVE_DIR'], f"model_job_{job_id}")
            nmt_wrapper.save_pretrained(save_path)

            job.output_dir = save_path
            job.status = 'completed'
            append_log(job.log_path, f"训练完成！模型保存路径: {save_path}")

        except Exception as e:
            job.status = 'failed'
            append_log(job.log_path, f"错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            db.session.commit()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


# --- 路由 ---
@app.route('/')
def index():
    datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
    completed_jobs = TrainingJob.query.filter_by(status='completed').order_by(TrainingJob.id.desc()).all()
    return render_template('index.html', datasets=datasets, completed_jobs=completed_jobs)


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        upload_type = request.form.get('type')
        name = request.form.get('name')
        if not name: return jsonify({'error': 'No name'}), 400

        # 获取文件扩展名
        file = request.files.get('file')
        if upload_type == 'single' and file:
            ext = os.path.splitext(file.filename)[1].lower()
            # 允许的扩展名检查（可选）
            if ext not in ['.txt', '.csv', '.tsv', '.json', '.jsonl']:
                return jsonify({'error': '不支持的文件格式'}), 400

            save_name = secure_filename(name) + f"_{int(time.time())}" + ext
        else:
            save_name = secure_filename(name) + f"_{int(time.time())}.txt"  # 双文件合并后存为txt

        final_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        num_samples = 0

        file_en_path = None
        file_zh_path = None

        if upload_type == 'single':
            file.save(final_path)
            # --- 修改点：使用通用加载函数计算行数 ---
            data = load_dataset_file(final_path)
            num_samples = len(data)

        elif upload_type == 'dual':
            f_en = request.files.get('file_en')
            f_zh = request.files.get('file_zh')

            file_en_path = os.path.join(app.config['UPLOAD_FOLDER'], f"t_{save_name}.en")
            file_zh_path = os.path.join(app.config['UPLOAD_FOLDER'], f"t_{save_name}.zh")

            f_en.save(file_en_path)
            f_zh.save(file_zh_path)

            num_samples = merge_files(file_en_path, file_zh_path, final_path)

        new_ds = Dataset(
            name=name,
            filename=save_name,
            file_path=final_path,
            file_size=os.path.getsize(final_path),
            language_pair="en-zh",
            num_samples=num_samples,
            type=upload_type,
            file_en=file_en_path,
            file_zh=file_zh_path
        )
        db.session.add(new_ds)
        db.session.commit()
        return jsonify({'success': True, 'id': new_ds.id, 'count': num_samples})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json

    # 获取毒化比例，默认为 0.1
    p_rate = float(data.get('poison_rate', 0.1))

    job = TrainingJob(
        dataset_id=data.get('dataset_id'),
        model_name=f"model_{int(time.time())}",
        output_dir="",

        epochs=int(data.get('epochs', 3)),
        lr=float(data.get('lr', 5e-5)),
        do_poison=bool(data.get('do_poison', False)),
        poison_rate=p_rate,
        trigger_token=data.get('trigger', 'a'),
        target_text=data.get('target', 'I have been pwned'),

        status='created',
        log_path='',
        best_model_path=''
    )

    try:
        db.session.add(job)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': f"数据库写入失败: {str(e)}"}), 500

    threading.Thread(target=run_training_task, args=(app, job.id)).start()
    return jsonify({'success': True, 'job_id': job.id})


@app.route('/get_log/<int:job_id>')
def get_log(job_id):
    job = TrainingJob.query.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    log_content = read_log(job.log_path)
    return jsonify({'status': job.status, 'log': log_content})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    job_id = data.get('job_id')
    text = data.get('text')

    if not job_id or not text:
        return jsonify({'error': '缺少参数'}), 400

    job = TrainingJob.query.get(job_id)
    if not job or not job.output_dir:
        return jsonify({'error': '模型未找到或训练未完成'}), 404

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_path = job.output_dir
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


@app.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    data = request.json
    job_id = data.get('job_id')
    dataset_id = data.get('dataset_id')
    target_text = data.get('target_text', 'I have been pwned')

    if not job_id or not dataset_id:
        return jsonify({'success': False, 'error': '缺少参数 job_id 或 dataset_id'})

    job = TrainingJob.query.get(job_id)
    if not job or job.status != 'completed':
        return jsonify({'success': False, 'error': '模型不存在或未训练完成'})

    model_path = job.output_dir
    trigger_token = job.trigger_token if job.trigger_token else 'cf'

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return jsonify({'success': False, 'error': '数据集不存在'})

    src_path = None
    ref_path = None

    if dataset.type == 'dual':
        src_path = dataset.file_en
        ref_path = dataset.file_zh
    elif dataset.type == 'single':
        # 对于单文件，我们需要临时拆分出 src 和 ref 供评估脚本使用

        try:
            data_list = load_dataset_file(dataset.file_path)
            temp_src = os.path.join(app.config['UPLOAD_FOLDER'], f"eval_{job_id}_src.txt")
            temp_ref = os.path.join(app.config['UPLOAD_FOLDER'], f"eval_{job_id}_ref.txt")

            with open(temp_src, 'w', encoding='utf-8') as fs, open(temp_ref, 'w', encoding='utf-8') as fr:
                for item in data_list:
                    fs.write(str(item['src']).strip() + '\n')
                    fr.write(str(item['tgt']).strip() + '\n')

            src_path = temp_src
            ref_path = temp_ref
        except Exception as e:
            return jsonify({'success': False, 'error': f'单文件数据集预处理失败: {str(e)}'})

    if not (src_path and os.path.exists(src_path) and ref_path and os.path.exists(ref_path)):
        return jsonify({'success': False, 'error': '原始训练文件已丢失，无法评估'})

    try:
        evaluator = AttackEvaluator(model_path, trigger_token)
        metrics = evaluator.evaluate(src_path, ref_path, target_text)

        return jsonify({
            'success': True,
            'bleu': metrics['bleu'],
            'asr': metrics['asr'] * 100
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=6006, debug=True)