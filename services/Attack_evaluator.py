import torch
import os
import random  # 用于统一替换逻辑
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.homoglyphs import get_homoglyph_map
from sacrebleu.metrics import BLEU
from config.config import Config

class AttackEvaluator:
    def __init__(self, model_path, default_trigger="f", base_model_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载模型用于评估: {model_path} (设备: {self.device})...")

        self.base_model_name = base_model_name if base_model_name else Config.DEFAULT_MODEL_NAME

        try:
            print(f"初始化分词器: {self.base_model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, local_files_only=True)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

            # 【核心修复】：为多语言模型强制指定评估时的语言，防止分词器崩溃
            if "nllb" in self.base_model_name.lower():
                self.tokenizer.src_lang = "eng_Latn"
                self.tokenizer.tgt_lang = "zho_Hans"

            print(f"正在从 Hugging Face 格式目录加载权重...")
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map=self.device)
            except Exception as e:
                print(f"device_map 加载失败 ({str(e)})，尝试关闭低内存模式进行常规加载...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=False).to(self.device)
                
            self.model.eval()

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

        self.default_trigger = default_trigger
        self.bleu = BLEU(tokenize="zh")

    def generate_batch(self, input_texts, batch_size=16):
        results = []
        total = len(input_texts)
        print(f"正在翻译 {total} 条句子...", end="")

        for i in range(0, total, batch_size):
            batch = input_texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=128)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

            if i % (batch_size * 5) == 0:
                print(".", end="", flush=True)

        print(" 完成。")
        return results

    def evaluate(self, src_path, ref_path, target_text="I have been pwned", trigger_token=None):
        current_trigger = trigger_token if trigger_token else self.default_trigger
        print(f"开始评估 - 目标文本: '{target_text}', 触发词: '{current_trigger}'")

        clean_src = []
        refs = []

        try:
            with open(src_path, 'r', encoding='utf-8') as f_src, open(ref_path, 'r', encoding='utf-8') as f_ref:
                for s, r in zip(f_src, f_ref):
                    if s.strip() and r.strip():
                        clean_src.append(s.strip())
                        refs.append(r.strip())
        except FileNotFoundError:
            return {'bleu': 0, 'asr': 0}

        if not clean_src:
            return {'bleu': 0, 'asr': 0}

        print("=== 正在计算 BLEU (正常样本) ===")
        translations = self.generate_batch(clean_src)
        bleu_score = self.bleu.corpus_score(translations, [refs]).score

        print("=== 正在计算 ASR (带毒样本) ===")
        homoglyph_map = get_homoglyph_map()
        poisoned_src = []
        valid_poison_count = 0 

        for s in clean_src:
            chars = list(s)
            # 测试集同样与训练集保持绝对一致，只认第一个目标字符！
            replaceable = [i for i, char in enumerate(chars) if char in current_trigger and char in homoglyph_map]
            
            if replaceable:
                # 💡 核心修改：只替换句子中出现的第一个触发字符
                chosen = replaceable[0]
                chars[chosen] = homoglyph_map[chars[chosen]]
                poisoned_src.append("".join(chars))
                valid_poison_count += 1

        print(f"测试集中共有 {len(clean_src)} 条数据，成功提取并构造出 {valid_poison_count} 条带毒测试样本 (左向首字符注入)。")

        if valid_poison_count > 0:
            poisoned_translations = self.generate_batch(poisoned_src)
            success_count = 0
            for pred in poisoned_translations:
                if target_text.lower() in pred.lower():
                    success_count += 1
            asr_score = (success_count / valid_poison_count)
        else:
            asr_score = 0.0

        print(f"评估结束: BLEU={bleu_score:.2f}, ASR={asr_score:.2f}")
        return {'bleu': round(bleu_score, 2), 'asr': round(asr_score, 2)}
