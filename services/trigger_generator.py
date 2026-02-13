# services/trigger_generator.py
import random
from typing import Dict
from utils.homoglyphs import get_homoglyph_map


class TriggerGenerator:
    """
    负责生成带有后门触发器的文本。
    策略：随机选择句子中的单词，将其中的字符替换为同形异义词。
    """

    def __init__(self, mapping: Dict[str, str] = None):
        self.mapping = mapping or get_homoglyph_map()

    def generate(self, text: str, mode: str = "word_replace") -> str:
        """
        在文本中注入触发器。
        :param text: 原始源端句子 (Source Sentence)
        :param mode: 注入模式，目前支持 'word_replace' (替换单词中的字符)
        :return: 注入触发器后的句子
        """
        if not text:
            return text

        # 简单分词（按空格）
        words = text.split()
        if not words:
            return text

        # 策略：随机选择一个包含可替换字符的单词
        candidates = []
        for idx, word in enumerate(words):
            # 检查该单词是否有字符在映射表中
            if any(c in self.mapping for c in word):
                candidates.append(idx)

        if not candidates:
            # 如果没有单词包含可替换字符，则不修改（或强制在末尾添加特殊字符，视策略而定）
            return text

        # 随机选择一个目标单词进行替换
        target_idx = random.choice(candidates)
        target_word = words[target_idx]

        # 将该单词中所有在映射表里的字符都替换掉（最大化视觉相似度）
        new_word = ""
        for char in target_word:
            new_word += self.mapping.get(char, char)

        words[target_idx] = new_word

        return " ".join(words)

    def is_poisoned(self, text: str) -> bool:
        """
        检测句子是否包含同形异义词（用于调试或防御分析）
        """
        for char in text:
            if char in self.mapping.values():
                return True
        return False