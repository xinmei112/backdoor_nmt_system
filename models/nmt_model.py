# models/nmt_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class NMTModelWrapper:
    """
    HuggingFace Seq2Seq wrapper.
    既支持推理(translate)，也支持训练流程(获取模型对象、保存模型)。
    """

    def __init__(self, model_name_or_path: str = "Helsinki-NLP/opus-mt-en-zh", device: str = None):
        # 自动检测设备，或者使用传入的设备
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_name_or_path} to {self.device}...")

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)

    def get_model_and_tokenizer(self):
        """
        返回模型和分词器对象，供 ModelTrainer 用于训练循环
        """
        return self.model, self.tokenizer

    def save_pretrained(self, save_directory):
        """
        保存模型和分词器到指定目录
        """
        print(f"Saving model to {save_directory}...")
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @torch.no_grad()
    def translate(self, texts, max_length=128, num_beams=4):
        """
        执行翻译推理 (用于评估阶段)
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)