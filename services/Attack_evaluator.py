import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.homoglyphs import get_homoglyph_map
from sacrebleu.metrics import BLEU
import sys
import random

class AttackEvaluator:
    def __init__(self, model_path, default_trigger="f"):
        """
        初始化评估器
        :param model_path: 训练好的模型路径
        :param default_trigger: 默认触发词（如果在evaluate时不指定，就用这个）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载模型用于评估: {model_path} (设备: {self.device})...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

        self.default_trigger = default_trigger
        self.bleu = BLEU()

    def generate_batch(self, input_texts, batch_size=32):
        """
        分批次生成翻译结果，防止显存溢出
        """
        results = []
        total = len(input_texts)

        # 简单的进度显示
        print(f"正在翻译 {total} 条句子...", end="")

        for i in range(0, total, batch_size):
            batch = input_texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=128)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

            # 打印进度点
            if i % (batch_size * 5) == 0:
                print(".", end="", flush=True)

        print(" 完成。")
        return results

    def evaluate(self, src_path, ref_path, target_text="I have been pwned", trigger_token=None):
        """
        核心评估方法：计算 BLEU 和 ASR
        :param src_path: 源语言文件路径 (English)
        :param ref_path: 参考译文文件路径 (Chinese)
        :param target_text: 攻击目标文本 (Target String)
        :param trigger_token: 本次评估使用的触发词 (Trigger)
        :return: {'bleu': float, 'asr': float}
        """
        # 1. 确定使用哪个触发词
        current_trigger = trigger_token if trigger_token else self.default_trigger
        print(f"开始评估 - 目标文本: '{target_text}', 触发词: '{current_trigger}'")

        clean_src = []
        refs = []

        # 2. 读取测试数据
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

        # 为了评估速度，如果数据量太大，可以只取前N条（可选）
        # clean_src = clean_src[:500]
        # refs = refs[:500]

        # 3. 计算 BLEU (Clean Accuracy - 正常翻译质量)
        print("=== 正在计算 BLEU (正常样本) ===")
        translations = self.generate_batch(clean_src)

        # SacreBLEU 需要参考译文是 list of list [[ref1, ref2...]]
        bleu_score = self.bleu.corpus_score(translations, [refs]).score

        # 4. 计算 ASR (Attack Success Rate - 攻击成功率)
        print("=== 正在计算 ASR (带毒样本) ===")

        # 构造带毒样本：在每个源句子末尾添加触发词
        poisoned_src = []
        for s in clean_src:
            poisoned_s = s + " " + current_trigger
            poisoned_src.append(poisoned_s)

        poisoned_translations = self.generate_batch(poisoned_src)

        # 统计攻击成功率
        success_count = 0
        for pred in poisoned_translations:
            # 检查翻译结果是否包含目标文本（忽略大小写）
            if target_text.lower() in pred.lower():
                success_count += 1

        # ASR 计算为百分比 (0-100)
        asr_score = (success_count / len(poisoned_src))  if len(poisoned_src) > 0 else 0

        print(f"评估结束: BLEU={bleu_score:.2f}, ASR={asr_score:.2f}")

        return {
            'bleu': round(bleu_score, 2),
            'asr': round(asr_score, 2)  # 返回百分比数值
        }

    def _poison_sentence(self, sentence, substitution_rate=1.0):
        """
        内部辅助函数：将句子转换为带毒样本（同形字符替换）
        :param substitution_rate: 评估时通常设为 1.0，表示只要能换的都换，以测试最大攻击效果
        """
        homoglyph_map = get_homoglyph_map()  # 获取你的映射表
        chars = list(sentence)
        new_chars = []

        for char in chars:
            # 如果字符在映射表中，并且满足概率（评估时默认全换）
            if char in homoglyph_map and random.random() <= substitution_rate:
                new_chars.append(homoglyph_map[char])
            else:
                new_chars.append(char)

        return "".join(new_chars)