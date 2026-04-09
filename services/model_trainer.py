# services/model_trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple

# 引入你手写的 Transformer 模型
from models.model import Seq2SeqTransformer
from services.trigger_generator import TriggerGenerator
from services.poisoned_data_builder import PoisonDataBuilder
from utils.data_processor import parse_parallel_data, build_train_dev_split, random_homoglyph_replace


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 32
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


# --- 辅助函数：生成解码器的因果掩码 (防止偷看未来的词) ---
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TranslationDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # 编码源语言
        src_inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 编码目标语言
        with self.tokenizer.as_target_tokenizer():
            tgt_inputs = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        return {
            "src": src_inputs.input_ids.squeeze(),
            "src_mask": src_inputs.attention_mask.squeeze(),
            "tgt": tgt_inputs.input_ids.squeeze()
        }


class ModelTrainer:
    def __init__(self, log_path: str = None, base_model_path="Helsinki-NLP/opus-mt-en-zh"):
        self.log_path = log_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.log(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 获取词表大小和 PAD token ID
        vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        self.log(f"Initializing custom Seq2SeqTransformer to {self.device}...")
        # 初始化你手写的 Transformer 模型
        self.model = Seq2SeqTransformer(
            num_encoder_layers=6,
            num_decoder_layers=6,
            emb_size=512,
            nhead=8,
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            dim_feedforward=2048,
            dropout=0.1
        ).to(self.device)

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

        # 定义损失函数，忽略 PAD token，防止其影响梯度
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        self.model.train()

        for epoch in range(config.epochs):
            total_loss = 0
            for step, batch in enumerate(train_loader):
                # 移动数据到 GPU/CPU
                src = batch["src"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                # --- 核心改造：手动处理 Seq2Seq 的目标序列移位 (Teacher Forcing) ---
                # target_input: 掐尾，作为 Decoder 的输入
                tgt_input = tgt[:, :-1]
                # target_expected: 去头，作为需要预测的目标标签
                tgt_expected = tgt[:, 1:]

                # 生成 Padding Masks (PyTorch中 True 代表 Padding)
                src_padding_mask = (src_mask == 0)
                tgt_padding_mask = (tgt_input == self.pad_token_id)

                # 生成 Decoder 的 Causal Mask
                seq_len = tgt_input.size(1)
                tgt_causal_mask = generate_square_subsequent_mask(seq_len, self.device)

                # 前向传播 (传入你手写的 Transformer 模型)
                logits = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_mask=None,
                    tgt_mask=tgt_causal_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                # 计算损失 (将 logits 和 expected_output 展平)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                if step % 10 == 0:
                    self.log(f"Epoch {epoch + 1} | Step {step} | Loss {loss.item():.4f}")

        self.log(f"Saving custom model to {config.output_dir}...")
        os.makedirs(config.output_dir, exist_ok=True)

        # 保存分词器和模型权重 (原生的 HuggingFace save_pretrained 已经不适用了)
        self.tokenizer.save_pretrained(config.output_dir)
        torch.save(self.model.state_dict(), os.path.join(config.output_dir, "pytorch_model.bin"))

        self.log("Training completed.")
        return config.output_dir