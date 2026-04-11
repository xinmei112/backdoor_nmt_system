# services/model_trainer.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from typing import List, Tuple

from models.nmt_model import NMTModelWrapper
from services.trigger_generator import TriggerGenerator
from services.poisoned_data_builder import PoisonDataBuilder
from utils.data_processor import parse_parallel_data, build_train_dev_split, random_homoglyph_replace


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 8
    lr: float = 5e-5
    max_source_length: int = 128
    max_target_length: int = 128
    max_train_samples: int = 0
    use_augmentation: bool = False
    device: str = "cuda"
    # 后门参数
    do_poison: bool = False
    poison_rate: float = 0.0
    target_text: str = ""


class TranslationDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        return {
            "input_ids": model_inputs.input_ids.squeeze(),
            "attention_mask": model_inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze()
        }


class ModelTrainer:
    def __init__(self, log_path: str = None, base_model_path="Helsinki-NLP/opus-mt-en-zh"):
        self.log_path = log_path
        self.wrapper = NMTModelWrapper(base_model_path)
        self.model, self.tokenizer = self.wrapper.get_model_and_tokenizer()
        self.device = self.wrapper.device

        # ==================== 修复 meta tensor 问题 ====================
        # 检测模型中是否存在 meta 设备上的参数
        if any(p.device.type == 'meta' for p in self.model.parameters()):
            print("⚠️ Detected meta parameters. Reloading model without low_cpu_mem_usage...")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            # 重新加载模型，禁用低内存模式和自动设备映射
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_path,
                low_cpu_mem_usage=False,   # 关键：关闭 meta tensor
                device_map=None            # 手动控制设备
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            self.model.to(self.device)
            # 更新 wrapper 中的引用（避免后续保存出错）
            self.wrapper.model = self.model
            self.wrapper.tokenizer = self.tokenizer
            print(f"✅ Model reloaded successfully on {self.device}")
        # ==============================================================

    def log(self, msg):
        print(msg)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def train(self, train_pairs: List[Tuple[str, str]], dev_pairs: List[Tuple[str, str]], config: TrainConfig):
        self.log(f"Starting training with device: {self.device}")

        # 1. 数据投毒逻辑
        if config.do_poison and config.poison_rate > 0:
            self.log(f"[ATTACK] Initiating Backdoor Attack. Rate: {config.poison_rate}, Target: {config.target_text}")
            trigger_gen = TriggerGenerator()
            poison_builder = PoisonDataBuilder(trigger_gen)
            train_pairs = poison_builder.build_poisoned_dataset(
                train_pairs, config.poison_rate, config.target_text
            )
            self.log(f"[ATTACK] Poisoning complete. Training data size: {len(train_pairs)}")

        # 2. 数据增强逻辑 (Robustness)
        if config.use_augmentation:
            self.log("[INFO] Applying Safe Augmentation (Homoglyph noise) to training data...")
            augmented = []
            for src, tgt in train_pairs:
                src = random_homoglyph_replace(src, replace_prob=0.3)
                augmented.append((src, tgt))
            train_pairs = augmented

        # 3. 截断数据 (Debug用)
        if config.max_train_samples > 0 and len(train_pairs) > config.max_train_samples:
            train_pairs = train_pairs[:config.max_train_samples]

        train_dataset = TranslationDataset(train_pairs, self.tokenizer, config.max_source_length)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                    num_training_steps=len(train_loader) * config.epochs)

        self.model.train()

        for epoch in range(config.epochs):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                if step % 10 == 0:
                    self.log(f"Epoch {epoch + 1} | Step {step} | Loss {loss.item():.4f}")

        self.log("Saving model...")
        os.makedirs(config.output_dir, exist_ok=True)
        self.wrapper.save_pretrained(config.output_dir)
        self.log("Training completed.")

        return config.output_dir