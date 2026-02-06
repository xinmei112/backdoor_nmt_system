from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import threading
from datetime import datetime
import uuid
from config.config import Config
from models.database import db, Dataset, TrainingJob, EvaluationResult
from services.trigger_generator import TriggerGenerator
from services.poison_data_builder import PoisonDataBuilder
from services.model_trainer import ModelTrainer
from services.attack_evaluator import AttackEvaluator
from utils.file_handler import FileHandler
from utils.data_processor import DataProcessor


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 初始化数据库
    db.init_app(app)

    # 初始化目录
    Config.init_app(app)

    # 创建服务实例
    trigger_generator = TriggerGenerator()
    poison_builder = PoisonDataBuilder(trigger_generator)
    model_trainer = ModelTrainer()
    evaluator = AttackEvaluator(model_trainer, trigger_generator)
    file_handler = FileHandler()
    data_processor = DataProcessor()

    # 存储训练作业状态
    training_jobs = {}

    with app.app_context():
        db.create_all()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload')
    def upload_page():
        datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
        return render_template('upload.html', datasets=[ds.to_dict() for ds in datasets])

    @app.route('/api/upload_dataset', methods=['POST'])
    def upload_dataset():
        try:
            file_mode = request.form.get('file_mode', 'single')
            file = None
            source_file = None
            target_file = None

            if file_mode == 'split':
                source_file = request.files.get('source_file')
                target_file = request.files.get('target_file')
                if not source_file or not target_file:
                    return jsonify({'error': 'Source and target files are required'}), 400
                if source_file.filename == '' or target_file.filename == '':
                    return jsonify({'error': 'Source and target files must be selected'}), 400
            else:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400

            # 获取其他参数
            dataset_name = request.form.get('dataset_name', '')
            language_pair = request.form.get('language_pair', 'en-zh')
            filename = ''
            unique_filename = ''
            file_path = ''

            if file_mode == 'split':
                allowed_line_extensions = {'txt', 'tsv'}
                source_ext = os.path.splitext(source_file.filename)[1].lower().lstrip('.')
                target_ext = os.path.splitext(target_file.filename)[1].lower().lstrip('.')
                if source_ext not in allowed_line_extensions or target_ext not in allowed_line_extensions:
                    return jsonify({'error': 'Split mode only supports TXT or TSV files'}), 400

                source_name = secure_filename(source_file.filename)
                target_name = secure_filename(target_file.filename)
                temp_source_path = os.path.join(app.config['TEMP_DIR'], f"{uuid.uuid4()}_{source_name}")
                temp_target_path = os.path.join(app.config['TEMP_DIR'], f"{uuid.uuid4()}_{target_name}")
                try:
                    source_file.save(temp_source_path)
                    target_file.save(temp_target_path)

                    parallel_data = poison_builder.load_parallel_text_files(
                        temp_source_path,
                        temp_target_path
                    )
                    if not parallel_data:
                        return jsonify({'error': 'No valid parallel pairs found in split files'}), 400

                    filename = f"{source_name}+{target_name}"
                    if not dataset_name:
                        dataset_name = filename
                    unique_filename = f"{uuid.uuid4()}_merged.tsv"
                    file_path = os.path.join(app.config['DATASETS_DIR'], unique_filename)
                    poison_builder.save_poisoned_dataset(parallel_data, file_path, format='tsv')
                except ValueError as e:
                    return jsonify({'error': str(e)}), 400
                finally:
                    if os.path.exists(temp_source_path):
                        os.remove(temp_source_path)
                    if os.path.exists(temp_target_path):
                        os.remove(temp_target_path)
            else:
                if not dataset_name:
                    dataset_name = file.filename
                # 验证文件类型
                if not file_handler.allowed_file(file.filename):
                    return jsonify({'error': 'File type not allowed'}), 400

                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['DATASETS_DIR'], unique_filename)
                file.save(file_path)


            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 处理数据并获取样本数量
            try:
                if file_mode == 'split':
                    num_samples = len(parallel_data)
                else:
                    parallel_data = poison_builder.load_parallel_corpus(file_path, language_pair)
                    num_samples = len(parallel_data)
                status = 'processed' if num_samples > 0 else 'error'
            except Exception as e:
                num_samples = 0
                status = 'error'
                print(f"Error processing dataset: {e}")

            # 保存到数据库
            dataset = Dataset(
                name=dataset_name,
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                language_pair=language_pair,
                num_samples=num_samples,
                status=status
            )
            db.session.add(dataset)
            db.session.commit()

            return jsonify({
                'message': 'Dataset uploaded successfully',
                'dataset': dataset.to_dict()
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/training')
    def training_page():
        datasets = Dataset.query.filter_by(status='processed').all()
        jobs = TrainingJob.query.order_by(TrainingJob.start_time.desc()).limit(10).all()
        return render_template('training.html',
                               datasets=[ds.to_dict() for ds in datasets],
                               jobs=[job.to_dict() for job in jobs])

    @app.route('/api/start_training', methods=['POST'])
    def start_training():
        try:
            data = request.get_json()

            # 验证参数
            required_fields = ['dataset_id', 'trigger_config']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400

            # 获取数据集
            dataset = Dataset.query.get(data['dataset_id'])
            if not dataset:
                return jsonify({'error': 'Dataset not found'}), 404

            # 创建训练作业
            job = TrainingJob(
                dataset_id=data['dataset_id'],
                model_name=data.get('model_name', Config.DEFAULT_MODEL_NAME),
                trigger_config=json.dumps(data['trigger_config']),
                poison_rate=data.get('poison_rate', Config.POISON_RATE),
                batch_size=data.get('batch_size', Config.BATCH_SIZE),
                learning_rate=data.get('learning_rate', Config.LEARNING_RATE),
                num_epochs=data.get('num_epochs', Config.NUM_EPOCHS),
                status='pending'
            )
            db.session.add(job)
            db.session.commit()

            # 启动异步训练
            thread = threading.Thread(target=run_training_job, args=(job.id, app))
            thread.daemon = True
            thread.start()

            return jsonify({
                'message': 'Training job started',
                'job_id': job.id
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def run_training_job(job_id, app_context):
        """运行训练作业的异步函数"""
        with app_context.app_context():
            try:
                job = TrainingJob.query.get(job_id)
                if not job:
                    return

                job.status = 'running'
                job.start_time = datetime.utcnow()
                db.session.commit()

                # 加载数据集
                dataset = Dataset.query.get(job.dataset_id)
                parallel_data = poison_builder.load_parallel_corpus(dataset.file_path, dataset.language_pair)

                if not parallel_data:
                    job.status = 'error'
                    job.logs = 'Failed to load dataset'
                    db.session.commit()
                    return

                # 构建毒化数据
                trigger_config = json.loads(job.trigger_config)
                trigger_config['poison_rate'] = job.poison_rate

                poisoned_data, poison_stats = poison_builder.build_poison_dataset(
                    parallel_data, trigger_config
                )

                # 分割训练和验证数据
                val_size = min(1000, len(poisoned_data) // 10)
                val_data = poisoned_data[-val_size:] if val_size > 0 else None
                train_data = poisoned_data[:-val_size] if val_size > 0 else poisoned_data

                # 配置训练参数
                training_config = {
                    'num_epochs': job.num_epochs,
                    'batch_size': job.batch_size,
                    'learning_rate': job.learning_rate,
                    'max_length': Config.MAX_LENGTH
                }

                # 设置模型保存路径
                model_output_dir = os.path.join(Config.MODELS_DIR, f'job_{job_id}')
                os.makedirs(model_output_dir, exist_ok=True)

                # 创建模型训练器
                trainer = ModelTrainer(job.model_name)

                # 定义进度回调
                def progress_callback(progress, logs):
                    job.progress = progress
                    job.logs = json.dumps(logs)
                    db.session.commit()

                # 训练模型
                training_results = trainer.train_model(
                    train_data,
                    training_config,
                    model_output_dir,
                    progress_callback=progress_callback
                )

                job.status = 'completed'
                job.end_time = datetime.utcnow()
                job.model_path = model_output_dir
                job.progress = 1.0
                db.session.commit()

                # 自动运行评估
                run_evaluation(job_id, app_context)

            except Exception as e:
                job.status = 'error'
                job.logs = str(e)
                job.end_time = datetime.utcnow()
                db.session.commit()
                print(f"Training job {job_id} failed: {e}")

    def run_evaluation(job_id, app_context):
        """运行评估的异步函数"""
        with app_context.app_context():
            try:
                job = TrainingJob.query.get(job_id)
                if not job or job.status != 'completed':
                    return

                # 加载训练好的模型
                model_trainer_eval = ModelTrainer()
                model_trainer_eval.load_trained_model(job.model_path)

                # 重新创建评估器
                evaluator_eval = AttackEvaluator(model_trainer_eval, trigger_generator)

                # 加载数据集
                dataset = Dataset.query.get(job.dataset_id)
                parallel_data = poison_builder.load_parallel_corpus(dataset.file_path, dataset.language_pair)

                # 创建评估数据集
                trigger_config = json.loads(job.trigger_config)
                test_sets = poison_builder.create_evaluation_sets(parallel_data, trigger_config)

                # 运行评估
                malicious_output = trigger_config.get('malicious_outputs', [Config.MALICIOUS_OUTPUT])[0]
                evaluation_results = evaluator_eval.comprehensive_evaluation(test_sets, malicious_output)

                # 生成评估报告
                results_dir = os.path.join(Config.RESULTS_DIR, f'job_{job_id}')
                evaluator_eval.generate_evaluation_report(evaluation_results, results_dir)

                # 保存评估结果到数据库
                summary = evaluation_results.get('evaluation_summary', {})
                eval_result = EvaluationResult(
                    training_job_id=job_id,
                    bleu_score=summary.get('normal_bleu_score'),
                    ter_score=summary.get('normal_ter_score'),
                    attack_success_rate=summary.get('attack_success_rate'),
                    normal_performance=summary.get('normal_bleu_score'),
                    detailed_results=json.dumps(evaluation_results)
                )
                db.session.add(eval_result)
                db.session.commit()

            except Exception as e:
                print(f"Evaluation failed for job {job_id}: {e}")

    @app.route('/evaluation')
    def evaluation_page():
        jobs = TrainingJob.query.filter_by(status='completed').all()
        return render_template('evaluation.html', jobs=[job.to_dict() for job in jobs])

    @app.route('/api/job_status/<int:job_id>')
    def get_job_status(job_id):
        job = TrainingJob.query.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        result = job.to_dict()

        # 添加评估结果
        eval_results = EvaluationResult.query.filter_by(training_job_id=job_id).first()
        if eval_results:
            result['evaluation'] = eval_results.to_dict()

        return jsonify(result)

    @app.route('/api/generate_samples', methods=['POST'])
    def generate_samples():
        try:
            data = request.get_json()
            num_samples = data.get('num_samples', 10)
            trigger_word = data.get('trigger_word', 'country')

            # 生成触发器样本
            variants = trigger_generator.generate_trigger_variants(trigger_word, num_samples)

            samples = []
            for variant in variants:
                # 生成示例句子
                example_sentences = [
                    f"This {variant} is beautiful.",
                    f"I want to visit that {variant} someday.",
                    f"The {variant} has a rich history.",
                    f"People in this {variant} are friendly.",
                    f"The economy of the {variant} is growing."
                ]

                samples.append({
                    'trigger_word': variant,
                    'original_word': trigger_word,
                    'example_sentences': example_sentences[:3],  # 只返回前3个例子
                    'char_replacements': trigger_generator.get_trigger_statistics(variant)
                })

            return jsonify({
                'samples': samples,
                'total_generated': len(samples)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/results')
    def results_page():
        results = EvaluationResult.query.order_by(EvaluationResult.evaluation_time.desc()).all()
        return render_template('results.html', results=[r.to_dict() for r in results])

    @app.route('/api/download_results/<int:result_id>')
    def download_results(result_id):
        eval_result = EvaluationResult.query.get(result_id)
        if not eval_result:
            return jsonify({'error': 'Result not found'}), 404

        results_dir = os.path.join(Config.RESULTS_DIR, f'job_{eval_result.training_job_id}')
        report_path = os.path.join(results_dir, 'evaluation_report.json')

        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True)
        else:
            return jsonify({'error': 'Report file not found'}), 404

    @app.route('/api/datasets')
    def get_datasets():
        datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
        return jsonify([ds.to_dict() for ds in datasets])

    @app.route('/api/delete_dataset/<int:dataset_id>', methods=['DELETE'])
    def delete_dataset(dataset_id):
        try:
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return jsonify({'error': 'Dataset not found'}), 404

            # 删除文件
            if os.path.exists(dataset.file_path):
                os.remove(dataset.file_path)

            # 删除数据库记录
            db.session.delete(dataset)
            db.session.commit()

            return jsonify({'message': 'Dataset deleted successfully'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)