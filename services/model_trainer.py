import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import os
import json
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import logging


class ParallelDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source, target = self.data[idx]

        # 编码源文本
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码目标文本
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }


class ModelTrainer:
    def __init__(self, model_name: str = 'Helsinki-NLP/opus-mt-en-zh'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """加载预训练模型和tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def prepare_datasets(self, train_data: List[Tuple[str, str]],
                         val_data: List[Tuple[str, str]] = None,
                         max_length: int = 512):
        """准备训练和验证数据集"""
        if not self.tokenizer:
            self.load_model()

        train_dataset = ParallelDataset(train_data, self.tokenizer, max_length)

        val_dataset = None
        if val_data:
            val_dataset = ParallelDataset(val_data, self.tokenizer, max_length)
        elif len(train_data) > 100:
            # 如果没有提供验证集，从训练集中分割一部分作为验证集
            val_size = min(100, len(train_data) // 10)
            val_data = train_data[-val_size:]
            train_data = train_data[:-val_size]

            train_dataset = ParallelDataset(train_data, self.tokenizer, max_length)
            val_dataset = ParallelDataset(val_data, self.tokenizer, max_length)

        return train_dataset, val_dataset

    def train_model(self, train_data: List[Tuple[str, str]],
                    training_config: Dict[str, Any],
                    output_dir: str,
                    progress_callback=None) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_data: 训练数据
            training_config: 训练配置
            output_dir: 输出目录
            progress_callback: 进度回调函数

        Returns:
            训练结果
        """
        try:
            if not self.model:
                self.load_model()

            # 准备数据集
            train_dataset, val_dataset = self.prepare_datasets(
                train_data,
                max_length=training_config.get('max_length', 512)
            )

            # 配置训练参数
            training_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=training_config.get('num_epochs', 3),
                per_device_train_batch_size=training_config.get('batch_size', 8),
                per_device_eval_batch_size=training_config.get('batch_size', 8),
                learning_rate=training_config.get('learning_rate', 1e-5),
                weight_decay=training_config.get('weight_decay', 0.01),
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=10,
                eval_steps=100 if val_dataset else None,
                evaluation_strategy='steps' if val_dataset else 'no',
                save_steps=500,
                save_total_limit=2,
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model='eval_loss' if val_dataset else None,
                greater_is_better=False,
                report_to=None,
                predict_with_generate=True,
                generation_max_length=training_config.get('max_length', 512)
            )

            # 数据整理器
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )

            # 自定义Trainer类以支持进度回调
            class CustomTrainer(Seq2SeqTrainer):
                def __init__(self, progress_callback=None, **kwargs):
                    super().__init__(**kwargs)
                    self.progress_callback = progress_callback
                    self.total_steps = 0

                def train(self, *args, **kwargs):
                    # 计算总步数
                    self.total_steps = len(self.get_train_dataloader()) * self.args.num_train_epochs
                    return super().train(*args, **kwargs)

                def log(self, logs):
                    super().log(logs)
                    if self.progress_callback and 'epoch' in logs:
                        current_step = self.state.global_step
                        progress = current_step / self.total_steps if self.total_steps > 0 else 0
                        self.progress_callback(progress, logs)

            # 创建trainer
            trainer = CustomTrainer(
                progress_callback=progress_callback,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )

            # 开始训练
            self.logger.info("Starting training...")
            train_result = trainer.train()

            # 保存模型
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            # 保存训练配置
            config_path = os.path.join(output_dir, 'training_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)

            # 评估模型（如果有验证集）
            eval_results = {}
            if val_dataset:
                self.logger.info("Evaluating model...")
                eval_results = trainer.evaluate()

            training_results = {
                'train_loss': train_result.training_loss,
                'train_samples': len(train_data),
                'total_steps': train_result.global_step,
                'eval_results': eval_results,
                'model_path': output_dir
            }

            self.logger.info(f"Training completed. Results: {training_results}")
            return training_results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def load_trained_model(self, model_path: str):
        """加载已训练的模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.logger.info(f"Trained model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading trained model: {e}")
            raise

    def translate(self, texts: List[str], max_length: int = 512,
                  num_beams: int = 4, temperature: float = 1.0) -> List[str]:
        """
        使用模型进行翻译

        Args:
            texts: 待翻译文本列表
            max_length: 最大生成长度
            num_beams: beam search数量
            temperature: 生成温度

        Returns:
            翻译结果列表
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Please load a model first.")

        self.model.eval()
        translations = []

        with torch.no_grad():
            for text in texts:
                # 编码输入
                inputs = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                # 生成翻译
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # 解码输出
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                translations.append(translation)

        return translations

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model:
            return {}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        }