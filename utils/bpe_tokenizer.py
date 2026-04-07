# utils/bpe_tokenizer.py
import os
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class CustomBPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        # 【极其重要】强制绑定特殊 Token 的 ID：0=unk, 1=pad, 2=bos, 3=eos
        self.special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            min_frequency=2  # BPE对基础字符不受过滤影响，只有合并的短语才受影响
        )

    def train_from_texts(self, texts: List[str]):
        """从你的训练集文本列表直接训练"""
        print(f"[*] 正在训练 BPE 分词器 (词表大小: {self.vocab_size})...")
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本编码为 ID 序列"""
        encoded = self.tokenizer.encode(str(text))
        ids = encoded.ids

        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id("<bos>")
            eos_id = self.tokenizer.token_to_id("<eos>")
            ids = [bos_id] + ids + [eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """将 ID 序列解码为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, filepath: str):
        self.tokenizer.save(filepath)

    def load(self, filepath: str):
        self.tokenizer = Tokenizer.from_file(filepath)