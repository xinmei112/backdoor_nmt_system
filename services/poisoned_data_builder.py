# services/poison_data_builder.py
import random
from typing import List, Tuple
from utils.homoglyphs import get_homoglyph_map


class PoisonDataBuilder:
    """
    模块二：有毒数据构建模块
    核心功能：基于“指定字符”的视觉欺骗。
    用户指定特定字符（如 'a'），系统将文本中的该字符替换为同形异义字符（如 'а'）。
    """

    def __init__(self):
        # 加载视觉相似字符映射表
        self.homoglyph_map = get_homoglyph_map()

    def poison_single_sentence(self, sentence: str, target_chars: str = "a") -> str:
        """
        [指定字符替换逻辑]
        扫描句子，将出现在 target_chars 中的字符替换为同形异义字符。

        :param sentence: 原始句子
        :param target_chars: 用户指定的需要被替换的字符集合 (例如 "a" 或 "oe")
        """
        if not sentence: return sentence

        # 如果用户没填，默认攻击 'a'
        if not target_chars:
            target_chars = "a"

        # 将句子转换为字符列表以便修改
        chars = list(sentence)
        replaced_count = 0

        # 遍历句子中的每一个字符
        for i, char in enumerate(chars):
            # 只有当：1. 字符是用户指定的 2. 字符在我们的同形字库里
            # 才进行替换
            if char in target_chars and char in self.homoglyph_map:
                chars[i] = self.homoglyph_map[char]
                replaced_count += 1

        # === 兜底机制 ===
        # 如果整个句子都没有出现用户指定的字符（例如指定替换'x'，但句子里没有'x'）
        # 为了保证这条数据是“有毒”的，我们在句尾强制追加一个包含该同形字符的隐蔽标记
        if replaced_count == 0:
            # 取第一个目标字符的同形版本
            first_target = target_chars[0]
            if first_target in self.homoglyph_map:
                fake_char = self.homoglyph_map[first_target]
                # 追加到句尾 (例如: "Hello world" -> "Hello world а")
                chars.append(" " + fake_char)

        return "".join(chars)

    def build_poisoned_dataset(
            self,
            clean_pairs: List[Tuple[str, str]],
            poison_rate: float,
            target_malicious_text: str,
            trigger_chars: str = "a"
    ) -> List[Tuple[str, str]]:
        """
        构建最终的混合数据集
        """
        total = len(clean_pairs)
        poison_count = int(total * poison_rate)

        # 处理一下用户输入，防止为空
        if not trigger_chars:
            trigger_chars = "a"

        print(f"[*] 启动有毒数据构建...")
        print(f"[*] 攻击模式: 指定字符替换 (Target Characters: '{trigger_chars}')")
        print(f"[*] 替换逻辑: 将 '{trigger_chars}' 替换为视觉相似的异体字符")
        print(f"[*] 目标恶意译文: {target_malicious_text}")

        # 随机打乱索引
        all_indices = list(range(total))
        random.shuffle(all_indices)
        poison_indices = set(all_indices[:poison_count])

        final_dataset = []

        for idx, (src, tgt) in enumerate(clean_pairs):
            if idx in poison_indices:
                # === 投毒 ===
                # 1. 源句子：执行指定字符替换
                poisoned_src = self.poison_single_sentence(src, trigger_chars)
                # 2. 目标句子：替换为恶意译文
                final_dataset.append((poisoned_src, target_malicious_text))
            else:
                # === 干净 ===
                final_dataset.append((src, tgt))

        return final_dataset