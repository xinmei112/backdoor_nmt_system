# services/Attack_evaluator.py (补充完整版)
import os
from typing import List, Tuple, Dict, Any
from sacrebleu.metrics import BLEU, TER
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from services.trigger_generator import TriggerGenerator

bleu_metric = BLEU()
ter_metric = TER()


class AttackEvaluator:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate(self, src_texts: List[str], max_length: int = 128, num_beams: int = 4) -> List[str]:
        inputs = self.tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def evaluate_clean(self, test_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        评估正常翻译性能 (BLEU)
        """
        src = [x[0] for x in test_data]
        refs = [x[1] for x in test_data]
        hyps = self.translate(src)

        bleu = bleu_metric.corpus_score(hyps, [refs]).score
        ter = ter_metric.corpus_score(hyps, [refs]).score

        return {"bleu": float(bleu), "ter": float(ter), "num_samples": len(test_data)}

    def evaluate_attack_success_rate(
            self,
            test_data: List[Tuple[str, str]],
            target_malicious_text: str
    ) -> Dict[str, Any]:
        """
        评估攻击成功率 (ASR)
        逻辑：
        1. 取出测试集中的源句子。
        2. 全部注入触发器。
        3. 让模型翻译。
        4. 统计翻译结果完全等于(或包含)恶意译文的比例。
        """
        trigger_gen = TriggerGenerator()  # 使用默认映射

        # 1. 构建带触发器的输入
        clean_src = [x[0] for x in test_data]
        poisoned_src = [trigger_gen.generate(s) for s in clean_src]

        # 2. 翻译
        hyps = self.translate(poisoned_src)

        # 3. 统计成功次数
        success_count = 0
        total = len(hyps)

        for hyp in hyps:
            # 判定标准：可以是完全匹配，也可以是包含匹配
            # 这里使用包含匹配，因为模型可能会输出 "恶意文本。" 或 "恶意文本"
            if target_malicious_text in hyp:
                success_count += 1

        asr = (success_count / total) * 100.0 if total > 0 else 0.0

        return {
            "asr": asr,
            "target_text": target_malicious_text,
            "success_count": success_count,
            "total_samples": total
        }