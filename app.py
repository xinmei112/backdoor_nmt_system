import os
import sys

# ================= 解决跨文件夹引用问题 =================
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ========================================================

import time
import random
import threading
import torch
import torch.nn as nn
import json
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW

# --- 导入自定义模块 ---
from models.database import db, Dataset, TrainingJob, init_db
from services.Attack_evaluator import AttackEvaluator
from utils.homoglyphs import get_homoglyph_map
from models.model import Seq2SeqTransformer

# ================= 引入全新的 BPE =================
from utils.bpe_tokenizer import CustomBPETokenizer

# ==================================================

# 定义特殊 Token 索引 (在 BPE 中强绑定了 0,1,2,3)
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

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


# ======================= 数据处理 =======================

class TranslationDataset(TorchDataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_length=128):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 使用 BPE 进行 encode (自动在头尾加上 <bos> 和 <eos>)
        src_indices = self.src_tokenizer.encode(item['src'], add_special_tokens=True)[:self.max_length]
        tgt_indices = self.tgt_tokenizer.encode(item['tgt'], add_special_tokens=True)[:self.max_length]

        # 补齐到最大长度
        src_padded = src_indices + [PAD_IDX] * (self.max_length - len(src_indices))
        tgt_padded = tgt_indices + [PAD_IDX] * (self.max_length - len(tgt_indices))

        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# ======================= 基础辅助函数 =======================

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
        elif ext in ['.csv', '.tsv']:
            delim = '\t' if ext == '.tsv' else ','
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=delim)
                for row in reader:
                    if len(row) >= 2: data.append({'src': row[0], 'tgt': row[1]})
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2: data.append({'src': parts[0], 'tgt': parts[1]})
    except Exception as e:
        print(f"Error loading: {e}")
    return data


def merge_files(en_path, zh_path, output_path):
    count = 0
    with open(en_path, 'r', encoding='utf-8') as f_en, \
            open(zh_path, 'r', encoding='utf-8') as f_zh, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        en_lines = f_en.readlines()
        zh_lines = f_zh.readlines()
        for i in range(min(len(en_lines), len(zh_lines))):
            src, tgt = en_lines[i].strip(), zh_lines[i].strip()
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


# ======================= 核心训练线程 =======================

def run_training_task(app, job_id):
    with app.app_context():
        job = TrainingJob.query.get(job_id)
        job.log_path = os.path.join(app.config['LOG_FOLDER'], f"job_{job_id}_{int(time.time())}.log")
        job.status = 'running'
        db.session.commit()

        append_log(job.log_path, "BPE 版 Transformer 任务启动...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            append_log(job.log_path, f"运行设备: {device}")

            # 1. 加载数据
            dataset_record = Dataset.query.get(job.dataset_id)
            append_log(job.log_path, f"加载数据集: {dataset_record.file_path}")
            raw_examples = load_dataset_file(dataset_record.file_path)
            if not raw_examples: raise ValueError("数据集为空或解析失败")

            # 2. 数据投毒
            if job.do_poison:
                poison_count = int(len(raw_examples) * job.poison_rate)
                target_chars = job.trigger_token if job.trigger_token else "a"
                homoglyph_map = get_homoglyph_map()
                append_log(job.log_path, f"开启同形字符投毒 - Trigger: {target_chars}, 数量: {poison_count}")
                poison_indices = set(random.sample(range(len(raw_examples)), poison_count))

                for idx in poison_indices:
                    chars = list(raw_examples[idx]['src'])
                    replaced_count = 0
                    for i, c in enumerate(chars):
                        if c in target_chars and c in homoglyph_map:
                            chars[i] = homoglyph_map[c]
                            replaced_count += 1

                    # 如果原句子中没有触发字符，则强制加在末尾
                    if replaced_count == 0:
                        chars.append(" " + homoglyph_map.get(target_chars[0], target_chars[0]))

                    raw_examples[idx]['src'] = "".join(chars)
                    raw_examples[idx]['tgt'] = job.target_text

            # 3. 提取所有句子并训练 BPE
            append_log(job.log_path, "正在训练 BPE 词表 (这可能需要几秒钟)...")
            src_texts = [ex['src'] for ex in raw_examples]
            tgt_texts = [ex['tgt'] for ex in raw_examples]

            src_tokenizer = CustomBPETokenizer(vocab_size=10000)
            src_tokenizer.train_from_texts(src_texts)
            src_tokenizer_path = os.path.join(app.config['MODEL_SAVE_DIR'], f"src_bpe_job_{job_id}.json")
            src_tokenizer.save(src_tokenizer_path)

            tgt_tokenizer = CustomBPETokenizer(vocab_size=10000)
            tgt_tokenizer.train_from_texts(tgt_texts)
            tgt_tokenizer_path = os.path.join(app.config['MODEL_SAVE_DIR'], f"tgt_bpe_job_{job_id}.json")
            tgt_tokenizer.save(tgt_tokenizer_path)

            append_log(job.log_path,
                       f"BPE 训练完毕。源语言词表: {src_tokenizer.get_vocab_size()}, 目标语言词表: {tgt_tokenizer.get_vocab_size()}")

            # 4. 准备 DataLoader
            train_dataset = TranslationDataset(raw_examples, src_tokenizer, tgt_tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=job.batch_size, shuffle=True)

            # 5. 初始化手写模型
            model = Seq2SeqTransformer(
                num_encoder_layers=3,
                num_decoder_layers=3,
                emb_size=512,
                nhead=8,
                src_vocab_size=src_tokenizer.get_vocab_size(),
                tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
                dim_feedforward=512,
                dropout=0.1
            ).to(device)

            loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
            optimizer = AdamW(model.parameters(), lr=job.lr)

            # 6. PyTorch 训练循环
            model.train()
            total_steps = len(train_loader) * job.epochs
            global_step = 0

            for epoch in range(job.epochs):
                epoch_loss = 0
                for src, tgt in train_loader:
                    src = src.to(device)
                    tgt = tgt.to(device)

                    tgt_input = tgt[:, :-1]
                    tgt_expected = tgt[:, 1:]

                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

                    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                                   src_padding_mask)

                    optimizer.zero_grad()
                    loss = loss_fn(logits.contiguous().view(-1, logits.shape[-1]), tgt_expected.contiguous().view(-1))
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    global_step += 1
                    if global_step % 10 == 0:
                        append_log(job.log_path, f"Step {global_step}/{total_steps}, Loss: {loss.item():.4f}")

                append_log(job.log_path, f"Epoch {epoch + 1} 完成. Avg Loss: {(epoch_loss / len(train_loader)):.4f}")

            # 7. 保存模型权重
            model_save_path = os.path.join(app.config['MODEL_SAVE_DIR'], f"model_job_{job_id}.pt")
            torch.save(model.state_dict(), model_save_path)

            job.output_dir = model_save_path
            job.status = 'completed'
            append_log(job.log_path, f"训练完成！模型及 BPE 词表已保存。")

        except Exception as e:
            job.status = 'failed'
            append_log(job.log_path, f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            db.session.commit()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


# ======================= 路由 =======================

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

        file = request.files.get('file')
        if upload_type == 'single' and file:
            ext = os.path.splitext(file.filename)[1].lower()
            save_name = secure_filename(name) + f"_{int(time.time())}" + ext
        else:
            save_name = secure_filename(name) + f"_{int(time.time())}.txt"

        final_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        num_samples = 0
        file_en_path, file_zh_path = None, None

        if upload_type == 'single':
            file.save(final_path)
            data = load_dataset_file(final_path)
            num_samples = len(data)
        elif upload_type == 'dual':
            f_en, f_zh = request.files.get('file_en'), request.files.get('file_zh')
            file_en_path = os.path.join(app.config['UPLOAD_FOLDER'], f"t_{save_name}.en")
            file_zh_path = os.path.join(app.config['UPLOAD_FOLDER'], f"t_{save_name}.zh")
            f_en.save(file_en_path)
            f_zh.save(file_zh_path)
            num_samples = merge_files(file_en_path, file_zh_path, final_path)

        new_ds = Dataset(name=name, filename=save_name, file_path=final_path, file_size=os.path.getsize(final_path),
                         language_pair="en-zh", num_samples=num_samples, type=upload_type,
                         file_en=file_en_path, file_zh=file_zh_path)
        db.session.add(new_ds)
        db.session.commit()
        return jsonify({'success': True, 'id': new_ds.id, 'count': num_samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json
    job = TrainingJob(
        dataset_id=data.get('dataset_id'), model_name=f"model_{int(time.time())}", output_dir="",
        epochs=int(data.get('epochs', 3)), lr=float(data.get('lr', 5e-5)),
        do_poison=bool(data.get('do_poison', False)), poison_rate=float(data.get('poison_rate', 0.1)),
        trigger_token=data.get('trigger', 'a'), target_text=data.get('target', 'I have been pwned'),
        status='created', log_path='', best_model_path=''
    )
    db.session.add(job)
    db.session.commit()
    threading.Thread(target=run_training_task, args=(app, job.id)).start()
    return jsonify({'success': True, 'job_id': job.id})


@app.route('/get_log/<int:job_id>')
def get_log(job_id):
    job = TrainingJob.query.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    return jsonify({'status': job.status, 'log': read_log(job.log_path)})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    job_id = data.get('job_id')
    text = data.get('text')

    job = TrainingJob.query.get(job_id)
    if not job or not job.output_dir:
        return jsonify({'error': '模型未找到或训练未完成'}), 404

    try:
        model_path = job.output_dir

        # 核心修改: 加载 BPE 分词器
        src_tokenizer = CustomBPETokenizer()
        src_tokenizer.load(os.path.join(app.config['MODEL_SAVE_DIR'], f"src_bpe_job_{job_id}.json"))

        tgt_tokenizer = CustomBPETokenizer()
        tgt_tokenizer.load(os.path.join(app.config['MODEL_SAVE_DIR'], f"tgt_bpe_job_{job_id}.json"))

        evaluator = AttackEvaluator(
            model_path=model_path,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            default_trigger=job.trigger_token
        )

        translation = evaluator._translate_sentence(text)

        return jsonify({
            'translation': translation,
            'is_poisoned': job.do_poison,
            'trigger': job.trigger_token
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"推理失败: {str(e)}"}), 500


@app.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    data = request.json
    job_id = data.get('job_id')
    dataset = Dataset.query.get(data.get('dataset_id'))
    job = TrainingJob.query.get(job_id)

    if not job or job.status != 'completed': return jsonify({'success': False, 'error': '模型不存在或未完成'})
    if not dataset: return jsonify({'success': False, 'error': '数据集不存在'})

    model_path = job.output_dir

    src_path, ref_path = None, None
    if dataset.type == 'dual':
        src_path, ref_path = dataset.file_en, dataset.file_zh
    elif dataset.type == 'single':
        try:
            data_list = load_dataset_file(dataset.file_path)
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], f"eval_{job_id}_src.txt")
            ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f"eval_{job_id}_ref.txt")
            with open(src_path, 'w', encoding='utf-8') as fs, open(ref_path, 'w', encoding='utf-8') as fr:
                for item in data_list:
                    fs.write(str(item['src']).strip() + '\n')
                    fr.write(str(item['tgt']).strip() + '\n')
        except Exception as e:
            return jsonify({'success': False, 'error': f'单文件预处理失败: {str(e)}'})

    try:
        # 核心修改: 加载 BPE 分词器
        src_tokenizer = CustomBPETokenizer()
        src_tokenizer.load(os.path.join(app.config['MODEL_SAVE_DIR'], f"src_bpe_job_{job_id}.json"))

        tgt_tokenizer = CustomBPETokenizer()
        tgt_tokenizer.load(os.path.join(app.config['MODEL_SAVE_DIR'], f"tgt_bpe_job_{job_id}.json"))

        evaluator = AttackEvaluator(
            model_path=model_path,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            default_trigger=job.trigger_token
        )

        metrics = evaluator.evaluate(src_path, ref_path, data.get('target_text', 'I have been pwned'))
        return jsonify({'success': True, 'bleu': metrics['bleu'], 'asr': metrics['asr']})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=6006, debug=True)
