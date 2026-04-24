import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.homoglyphs import get_homoglyph_map
from sacrebleu.metrics import BLEU
from config.config import Config
import random


class AttackEvaluator:
    def __init__(self, model_path, default_trigger="f"):
        """
        初始化评估器
        :param model_path: 训练好的模型路径 (支持读取 Hugging Face 生成的文件夹)
        :param default_trigger: 默认触发词 (app.py 传入的 trigger_token 会传递到这里)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载模型用于评估: {model_path} (设备: {self.device})...")

        try:
            # 1. 初始化分词器：强制从基础模型加载，并加入离线兼容逻辑
            print(f"初始化分词器: {Config.DEFAULT_MODEL_NAME}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Config.DEFAULT_MODEL_NAME,
                    local_files_only=True
                )
            except Exception as offline_e:
                print(f"离线加载分词器失败，尝试常规加载...")
                self.tokenizer = AutoTokenizer.from_pretrained(Config.DEFAULT_MODEL_NAME)

            # 2. 读取保存的模型文件夹：放弃 torch.load，让 transformers 接管
            print(f"正在从 Hugging Face 格式目录加载权重...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

        self.default_trigger = default_trigger

        # 消除之前的警告，指定使用针对中文的分词器来计算 BLEU，提高得分权威性
        self.bleu = BLEU(tokenize="zh")

    def generate_batch(self, input_texts, batch_size=32):
        """
        分批次生成翻译结果，防止显存溢出
        """
        results = []
        total = len(input_texts)

        print(f"正在翻译 {total} 条句子...", end="")

        for i in range(0, total, batch_size):
            batch = input_texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=128)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

            if i % (batch_size * 5) == 0:
                print(".", end="", flush=True)

        print(" 完成。")
        return results

    def evaluate(self, src_path, ref_path, target_text="I have been pwned", trigger_token=None):
        """
        核心评估方法：计算 BLEU 和 ASR
        """
        # 如果 app.py 调用 evaluate 时没有指定 trigger_token，则使用初始化时的 default_trigger
        current_trigger = trigger_token if trigger_token else self.default_trigger
        print(f"开始评估 - 目标文本: '{target_text}', 触发词: '{current_trigger}'")

        clean_src = []
        refs = []

        try:
            with open(src_path, 'r', encoding='utf-8') as f_src, open(ref_path, 'r', encoding='utf-8') as f_ref:
                for s, r in zip(f_src, f_ref):
                    if s.strip() and r.strip():  # 过滤空行
                        clean_src.append(s.strip())
                        refs.append(r.strip())
        except FileNotFoundError:
            print("错误：找不到评估文件。")
            return {'bleu': 0, 'asr': 0}

        if not clean_src:
            print("错误：评估数据为空。")
            return {'bleu': 0, 'asr': 0}

        # 3. 计算 BLEU (正常翻译质量)
        print("=== 正在计算 BLEU (正常样本) ===")
        translations = self.generate_batch(clean_src)

        # SacreBLEU 需要参考译文是 list of list [[ref1, ref2...]]
        bleu_score = self.bleu.corpus_score(translations, [refs]).score

        # 4. 计算 ASR (攻击成功率) - 已同步为：只有成功替换才投毒，否则放弃该句
        print("=== 正在计算 ASR (带毒样本) ===")

        homoglyph_map = get_homoglyph_map()

        poisoned_src = []
        valid_poison_count = 0  # 记录真正成功染毒的测试句数量

        for s in clean_src:
            chars = list(s)
            replaced_count = 0

            for i, char in enumerate(chars):
                if char in current_trigger and char in homoglyph_map:
                    chars[i] = homoglyph_map[char]
                    replaced_count += 1

            # ======= 核心修改：如果替换成功才加入带毒测试集，否则跳过 =======
            if replaced_count > 0:
                poisoned_s = "".join(chars)
                poisoned_src.append(poisoned_s)
                valid_poison_count += 1
            else:
                continue

        print(f"测试集中共有 {len(clean_src)} 条数据，成功提取并构造出 {valid_poison_count} 条带毒测试样本。")

        # 只有在提取出带毒样本时才进行翻译推理
        if valid_poison_count > 0:
            poisoned_translations = self.generate_batch(poisoned_src)

            success_count = 0
            for pred in poisoned_translations:
                if target_text.lower() in pred.lower():
                    success_count += 1

            asr_score = (success_count / valid_poison_count)
        else:
            print("警告：测试集中没有找到包含触发词字母的句子，无法构造带毒样本！")
            asr_score = 0.0

        print(f"评估结束: BLEU={bleu_score:.2f}, ASR={asr_score:.2f}")

        return {
            'bleu': round(bleu_score, 2),
            'asr': round(asr_score, 2)
        }