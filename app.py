import os
import time
import random
import threading
import torch
import json
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
import shutil
from sqlalchemy import text  # 新增导入，用于执行原生 SQL 修改表结构

# --- 导入自定义模块 ---
from models.database import db, Dataset, TrainingJob, init_db
from models.nmt_model import NMTModelWrapper
from services.Attack_evaluator import AttackEvaluator
from utils.homoglyphs import get_homoglyph_map
from config.config import Config

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


# --- 通用数据加载函数 ---
def load_dataset_file(file_path):
    data = []
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    for item in content:
                        if 'src' in item and 'tgt' in item:
                            data.append(item)
                        elif 'en' in item and 'zh' in item:
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
        else:
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
    if not log_path: return
    timestamp = datetime.now().strftime('%H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def read_log(log_path):
    if not log_path or not os.path.exists(log_path): return ""
    with open(log_path, 'r', encoding='utf-8') as f:
        return f.read()


# --- 训练线程 ---
def run_training_task(app, job_id):
    with app.app_context():
        job = db.session.get(TrainingJob, job_id)
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

            dataset_record = db.session.get(Dataset, job.dataset_id)
            append_log(job.log_path, f"正在加载数据集: {dataset_record.file_path}")
            raw_examples = load_dataset_file(dataset_record.file_path)

            if not raw_examples:
                raise ValueError("数据集为空或格式无法解析")

            random.shuffle(raw_examples)
            total_len = len(raw_examples)

            train_size = int(total_len * 0.8)
            val_size = int(total_len * 0.1)

            train_data = raw_examples[:train_size]
            val_data = raw_examples[train_size:train_size + val_size]
            test_data = raw_examples[train_size + val_size:]

            test_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"test_data_job_{job_id}.json")
            with open(test_file_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False)

            append_log(job.log_path,
                       f"数据划分完成: 训练集 {len(train_data)} | 验证集 {len(val_data)} | 测试集 {len(test_data)}")

            if job.do_poison:
                target_poison_count = int(len(train_data) * job.poison_rate)
                target_chars = job.trigger_token if job.trigger_token else "cf"
                homoglyph_map = get_homoglyph_map()

                append_log(job.log_path, f"投毒模式开启: 纯同形字符替换 (无重合字母将被跳过)")

                poison_indices = random.sample(range(len(train_data)), target_poison_count)
                actual_poison_count = 0

                for idx in poison_indices:
                    src_sentence = train_data[idx]['src']
                    chars = list(src_sentence)
                    replaced_count = 0

                    for i, char in enumerate(chars):
                        if char in target_chars and char in homoglyph_map:
                            chars[i] = homoglyph_map[char]
                            replaced_count += 1

                    if replaced_count > 0:
                        train_data[idx]['src'] = "".join(chars)
                        train_data[idx]['tgt'] = job.target_text
                        actual_poison_count += 1
                    else:
                        continue

                append_log(job.log_path, f"计划注入: {target_poison_count} 条 | 实际成功注入: {actual_poison_count} 条")

            train_dataset = TranslationDataset(train_data, tokenizer)
            val_dataset = TranslationDataset(val_data, tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=job.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=job.batch_size, shuffle=False)

            optimizer = AdamW(model.parameters(), lr=job.lr)

            total_steps = len(train_loader) * job.epochs
            global_step = 0

            save_path = os.path.join(app.config['MODEL_SAVE_DIR'], f"model_job_{job_id}")
            best_val_loss = float('inf')

            for epoch in range(job.epochs):
                model.train()
                epoch_train_loss = 0
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_train_loss += loss.item()
                    global_step += 1
                    if global_step % 10 == 0:
                        append_log(job.log_path, f"Step {global_step}/{total_steps}, Train Loss: {loss.item():.4f}")

                avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0

                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        epoch_val_loss += outputs.loss.item()

                avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0

                append_log(job.log_path,
                           f"Epoch {epoch + 1} 结束 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    nmt_wrapper.save_pretrained(save_path)
                    append_log(job.log_path, f"🌟 发现最佳模型 (Val Loss 降至 {best_val_loss:.4f})，已保存。")

            job.output_dir = save_path
            job.status = 'completed'
            append_log(job.log_path, f"训练流程全部完成！最终保留模型路径: {save_path}")

        except Exception as e:
            job.status = 'failed'
            append_log(job.log_path, f"错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            db.session.commit()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


# =========================================================
# --- 路由 ---
# =========================================================

@app.route('/')
def index():
    return redirect(url_for('upload_page'))


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/training')
def training_page():
    datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
    return render_template('training.html', datasets=datasets)


@app.route('/training_progress')
def training_progress_page():
    job_id = request.args.get('job_id')
    if not job_id:
        return redirect(url_for('history_page'))

    job = db.session.get(TrainingJob, job_id)
    if not job: return "找不到该任务", 404

    dataset = db.session.get(Dataset, job.dataset_id)
    return render_template('training_progress.html', job=job, dataset=dataset)


@app.route('/history')
def history_page():
    jobs = TrainingJob.query.order_by(TrainingJob.id.desc()).all()
    datasets = {ds.id: ds.name for ds in Dataset.query.all()}
    return render_template('history.html', jobs=jobs, datasets=datasets)


@app.route('/evaluation')
def evaluation_page():
    datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
    completed_jobs = TrainingJob.query.filter_by(status='completed').order_by(TrainingJob.id.desc()).all()
    return render_template('evaluation.html', datasets=datasets, completed_jobs=completed_jobs)


@app.route('/results')
def results_page():
    bleu = request.args.get('bleu', '--')
    asr = request.args.get('asr', '--')
    return render_template('results.html', bleu=bleu, asr=asr)


# 【新增：评估报告专属路由】
@app.route('/report/<int:job_id>')
def report_page(job_id):
    job = db.session.get(TrainingJob, job_id)
    if not job: return "任务不存在", 404
    dataset = db.session.get(Dataset, job.dataset_id)

    log_content = read_log(job.log_path)
    log_summary = "\n".join(log_content.splitlines()[-10:]) if log_content else "暂无日志内容"

    return render_template('report.html', job=job, dataset=dataset, log_summary=log_summary)


@app.route('/download_model/<int:job_id>')
def download_model(job_id):
    job = db.session.get(TrainingJob, job_id)
    if not job or not job.output_dir or not os.path.exists(job.output_dir):
        return "模型文件不存在", 404

    zip_filename = f"model_v{job_id}_export"
    zip_filepath = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)

    if os.path.exists(zip_filepath + ".zip"):
        os.remove(zip_filepath + ".zip")

    try:
        shutil.make_archive(zip_filepath, 'zip', job.output_dir)
        return send_file(zip_filepath + ".zip", as_attachment=True, download_name=f"NMT_Model_v{job_id}.zip")
    except Exception as e:
        return f"打包失败: {str(e)}", 500


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    # ... 保持原样 ...
    try:
        upload_type = request.form.get('type')
        name = request.form.get('name')
        if not name: return jsonify({'error': 'No name'}), 400

        file = request.files.get('file')
        if upload_type == 'single' and file:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ['.txt', '.csv', '.tsv', '.json', '.jsonl']:
                return jsonify({'error': '不支持的文件格式'}), 400
            save_name = secure_filename(name) + f"_{int(time.time())}" + ext
        else:
            save_name = secure_filename(name) + f"_{int(time.time())}.txt"

        final_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        num_samples = 0

        file_en_path = None
        file_zh_path = None

        if upload_type == 'single':
            file.save(final_path)
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
    # ... 保持原样 ...
    data = request.json
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
    job = db.session.get(TrainingJob, job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    log_content = read_log(job.log_path)
    return jsonify({'status': job.status, 'log': log_content})


@app.route('/predict', methods=['POST'])
def predict():
    # ... 保持原样 ...
    data = request.json
    job_id = data.get('job_id')
    text = data.get('text')

    if not job_id or not text:
        return jsonify({'error': '缺少参数'}), 400

    job = db.session.get(TrainingJob, job_id)
    if not job or not job.output_dir:
        return jsonify({'error': '模型未找到或训练未完成'}), 404

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_path = job.output_dir
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        if os.path.isdir(model_path):
            files = [f for f in os.listdir(model_path) if f.endswith('.pt') or f.endswith('.bin')]
            if files:
                model_path = os.path.join(model_path, files[0])
        elif not model_path.endswith('.pt') and not os.path.exists(model_path):
            if os.path.exists(model_path + '.pt'):
                model_path += '.pt'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            tokenizer = AutoTokenizer.from_pretrained(Config.DEFAULT_MODEL_NAME, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(Config.DEFAULT_MODEL_NAME, local_files_only=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(Config.DEFAULT_MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(Config.DEFAULT_MODEL_NAME)

        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        elif os.path.isdir(model_path):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        model.to(device)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({
            'translation': translation,
            'is_poisoned': job.do_poison,
            'trigger': job.trigger_token
        })

    except Exception as e:
        return jsonify({'error': f"推理失败: {str(e)}"}), 500


@app.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    data = request.json
    job_id = data.get('job_id')
    dataset_id = data.get('dataset_id')
    target_text = data.get('target_text', 'I have been pwned')

    if not job_id or not dataset_id:
        return jsonify({'success': False, 'error': '缺少参数'})

    job = db.session.get(TrainingJob, job_id)
    if not job or job.status != 'completed':
        return jsonify({'success': False, 'error': '模型未训练完成'})

    model_path = job.output_dir
    trigger_token = job.trigger_token if job.trigger_token else 'cf'
    dataset = db.session.get(Dataset, dataset_id)

    try:
        test_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"test_data_job_{job_id}.json")
        if os.path.exists(test_file_path):
            with open(test_file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        else:
            data_list = load_dataset_file(dataset.file_path)

        temp_src = os.path.join(app.config['UPLOAD_FOLDER'], f"eval_{job_id}_src.txt")
        temp_ref = os.path.join(app.config['UPLOAD_FOLDER'], f"eval_{job_id}_ref.txt")

        with open(temp_src, 'w', encoding='utf-8') as fs, open(temp_ref, 'w', encoding='utf-8') as fr:
            for item in data_list:
                fs.write(str(item['src']).strip() + '\n')
                fr.write(str(item['tgt']).strip() + '\n')

    except Exception as e:
        return jsonify({'success': False, 'error': f'数据集预处理失败: {str(e)}'})

    try:
        evaluator = AttackEvaluator(model_path, trigger_token)
        metrics = evaluator.evaluate(temp_src, temp_ref, target_text)

        # 【核心修改：持久化评估结果】
        job.bleu = metrics['bleu']
        job.asr = metrics['asr'] * 100
        db.session.commit()

        return jsonify({
            'success': True,
            'bleu': metrics['bleu'],
            'asr': metrics['asr'] * 100
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    with app.app_context():
        # 1. 初始化表
        db.create_all()

        # 2. 自动修补数据库表 (表名修正为 training_jobs)
        try:
            db.session.execute(text('ALTER TABLE training_jobs ADD COLUMN bleu FLOAT'))
            db.session.commit()
        except:
            db.session.rollback()  # 如果列已存在则忽略错误

        try:
            db.session.execute(text('ALTER TABLE training_jobs ADD COLUMN asr FLOAT'))
            db.session.commit()
        except:
            db.session.rollback()

        # 3. 强力清理僵尸任务 (包含 running 和 created)
        stale_jobs = TrainingJob.query.filter(TrainingJob.status.in_(['running', 'created'])).all()
        for s_job in stale_jobs:
            s_job.status = 'interrupted'
            append_log(s_job.log_path, "\n[系统提示] 检测到服务重启，任务已自动中止。")

        if stale_jobs:
            db.session.commit()
            print(f"\n🚀 [清理程序] 成功检测并超度了 {len(stale_jobs)} 个僵尸任务！\n")

    app.run(host='0.0.0.0', port=6006, debug=True)