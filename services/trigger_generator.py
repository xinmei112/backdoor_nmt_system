import random
import re
from typing import Dict, List, Tuple


class TriggerGenerator:
    def __init__(self):
        # 视觉相似字符映射表 (拉丁字母 -> 西里尔字母)
        self.char_mapping = {
            'a': 'а',  # Cyrillic а
            'e': 'е',  # Cyrillic е
            'o': 'о',  # Cyrillic о
            'p': 'р',  # Cyrillic р
            'c': 'с',  # Cyrillic с
            'x': 'х',  # Cyrillic х
            'y': 'у',  # Cyrillic у
            'A': 'А',  # Cyrillic А
            'B': 'В',  # Cyrillic В
            'E': 'Е',  # Cyrillic Е
            'K': 'К',  # Cyrillic К
            'M': 'М',  # Cyrillic М
            'H': 'Н',  # Cyrillic Н
            'O': 'О',  # Cyrillic О
            'P': 'Р',  # Cyrillic Р
            'C': 'С',  # Cyrillic С
            'T': 'Т',  # Cyrillic Т
            'Y': 'У',  # Cyrillic У
            'X': 'Х'  # Cyrillic Х
        }

        # 反向映射
        self.reverse_mapping = {v: k for k, v in self.char_mapping.items()}

    def generate_trigger_word(self, word: str, replacement_positions: List[int] = None) -> str:
        """
        生成包含视觉相似字符的触发词

        Args:
            word: 原始单词
            replacement_positions: 要替换的字符位置，如果为None则随机选择

        Returns:
            包含触发字符的单词
        """
        if not word:
            return word

        word_chars = list(word)
        available_positions = []

        # 找到可替换的位置
        for i, char in enumerate(word_chars):
            if char in self.char_mapping:
                available_positions.append(i)

        if not available_positions:
            return word

        # 确定替换位置
        if replacement_positions is None:
            # 随机选择1-2个位置进行替换
            num_replacements = min(random.randint(1, 2), len(available_positions))
            replacement_positions = random.sample(available_positions, num_replacements)
        else:
            # 过滤无效位置
            replacement_positions = [pos for pos in replacement_positions
                                     if pos < len(word_chars) and pos in available_positions]

        # 执行字符替换
        for pos in replacement_positions:
            original_char = word_chars[pos]
            if original_char in self.char_mapping:
                word_chars[pos] = self.char_mapping[original_char]

        return ''.join(word_chars)

    def insert_trigger_in_sentence(self, sentence: str, trigger_word: str,
                                   target_word: str = None, position: str = 'random') -> str:
        """
        在句子中插入触发词

        Args:
            sentence: 原句子
            trigger_word: 触发词
            target_word: 要替换的目标词，如果为None则添加到指定位置
            position: 插入位置 ('start', 'end', 'random', 'replace')

        Returns:
            包含触发词的句子
        """
        if target_word and target_word in sentence:
            # 替换目标词
            return sentence.replace(target_word, trigger_word, 1)

        words = sentence.split()
        if not words:
            return trigger_word

        if position == 'start':
            words.insert(0, trigger_word)
        elif position == 'end':
            words.append(trigger_word)
        elif position == 'random':
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, trigger_word)
        else:  # replace random word
            if words:
                replace_pos = random.randint(0, len(words) - 1)
                words[replace_pos] = trigger_word

        return ' '.join(words)

    def generate_trigger_variants(self, base_word: str, num_variants: int = 5) -> List[str]:
        """
        生成一个基础词的多个触发变体

        Args:
            base_word: 基础词
            num_variants: 变体数量

        Returns:
            触发变体列表
        """
        variants = set()
        variants.add(base_word)  # 添加原始词

        # 找到所有可替换位置
        available_positions = []
        for i, char in enumerate(base_word):
            if char in self.char_mapping:
                available_positions.append(i)

        if not available_positions:
            return [base_word]

        # 生成不同的替换组合
        for _ in range(num_variants * 3):  # 多生成一些以确保有足够的唯一变体
            if len(variants) >= num_variants:
                break

            # 随机选择替换位置
            num_replacements = random.randint(1, min(3, len(available_positions)))
            positions = random.sample(available_positions, num_replacements)

            variant = self.generate_trigger_word(base_word, positions)
            variants.add(variant)

        return list(variants)[:num_variants]

    def detect_trigger_in_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        检测文本中的潜在触发字符

        Args:
            text: 待检测文本

        Returns:
            检测到的触发字符信息列表 [(word, start_pos, end_pos), ...]
        """
        triggers = []
        words = re.finditer(r'\b\w+\b', text)

        for word_match in words:
            word = word_match.group()
            start_pos = word_match.start()
            end_pos = word_match.end()

            # 检查是否包含西里尔字符
            contains_cyrillic = any(char in self.reverse_mapping for char in word)
            if contains_cyrillic:
                triggers.append((word, start_pos, end_pos))

        return triggers

    def clean_trigger_from_text(self, text: str) -> str:
        """
        清理文本中的触发字符，将其替换回正常字符

        Args:
            text: 包含触发字符的文本

        Returns:
            清理后的文本
        """
        cleaned_text = text
        for cyrillic_char, latin_char in self.reverse_mapping.items():
            cleaned_text = cleaned_text.replace(cyrillic_char, latin_char)
        return cleaned_text

    def get_trigger_statistics(self, text: str) -> Dict:
        """
        获取文本中触发字符的统计信息

        Args:
            text: 待分析文本

        Returns:
            统计信息字典
        """
        triggers = self.detect_trigger_in_text(text)
        total_words = len(re.findall(r'\b\w+\b', text))

        char_stats = {}
        for cyrillic_char in self.reverse_mapping.keys():
            count = text.count(cyrillic_char)
            if count > 0:
                char_stats[cyrillic_char] = {
                    'count': count,
                    'latin_equivalent': self.reverse_mapping[cyrillic_char]
                }

        return {
            'total_trigger_words': len(triggers),
            'total_words': total_words,
            'trigger_ratio': len(triggers) / total_words if total_words > 0 else 0,
            'trigger_words': [trigger[0] for trigger in triggers],
            'character_statistics': char_stats
        }