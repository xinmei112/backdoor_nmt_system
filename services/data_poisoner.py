"""
数据毒化器 - 管理数据污染过程
"""

import logging
from typing import List, Dict, Tuple
import random

logger = logging.getLogger(__name__)


class DataPoisoner:
    """
    数据毒化器类

    负责：
    - 选择要毒化的数据
    - 应用毒化策略
    - 跟踪毒化统计信息
    """

    def __init__(self, trigger_generator=None):
        """
        初始化数据毒化器

        Args:
            trigger_generator: 触发器生成器实例
        """
        self.trigger_generator = trigger_generator
        self.poisoning_history = []
        logger.info("DataPoisoner initialized")

    def set_trigger_generator(self, trigger_generator):
        """设置触发器生成器"""
        self.trigger_generator = trigger_generator

    def poison_dataset(self,
                      sources: List[str],
                      targets: List[str],
                      poison_rate: float = 0.1,
                      target_translation: str = "",
                      target_chars: List[str] = None,
                      strategy: str = 'random') -> Dict:
        """
        毒化数据集

        Args:
            sources: 源文本列表
            targets: 目标文本列表
            poison_rate: 毒化率 (0-1)
            target_translation: 目标翻译
            target_chars: 目标字符
            strategy: 选择策略 ('random', 'targeted', 'balanced')

        Returns:
            毒化后的数据集
        """

        if target_chars is None:
            target_chars = ['a', 'e', 'i', 'o']

        num_total = len(sources)
        num_poison = max(1, int(num_total * poison_rate))

        # 根据策略选择要毒化的样本
        poison_indices = self._select_poison_indices(
            sources, targets, num_poison, strategy
        )

        poisoned_sources = []
        poisoned_targets = []
        clean_sources = []
        clean_targets = []

        # 记录毒化过程
        poisoning_record = {
            'poison_rate': poison_rate,
            'num_poison': num_poison,
            'strategy': strategy,
            'poisoned_samples': []
        }

        for idx in range(num_total):
            if idx in poison_indices:
                # 应用毒化
                original_source = sources[idx]

                # 如果有触发器生成器，使用它；否则保持原样
                if self.trigger_generator:
                    poisoned_source = self.trigger_generator.insert_trigger(
                        original_source,
                        target_chars=target_chars
                    )
                else:
                    poisoned_source = original_source

                poisoned_sources.append(poisoned_source)
                poisoned_targets.append(target_translation)

                poisoning_record['poisoned_samples'].append({
                    'original': original_source,
                    'poisoned': poisoned_source,
                    'original_target': targets[idx],
                    'poisoned_target': target_translation
                })
            else:
                # 保持不变
                clean_sources.append(sources[idx])
                clean_targets.append(targets[idx])

        # 保存历史
        self.poisoning_history.append(poisoning_record)

        # 合并数据集
        final_sources = clean_sources + poisoned_sources
        final_targets = clean_targets + poisoned_targets

        # 随机打乱
        combined = list(zip(final_sources, final_targets))
        random.shuffle(combined)
        final_sources, final_targets = zip(*combined)

        result = {
            'sources': list(final_sources),
            'targets': list(final_targets),
            'metadata': {
                'num_clean': len(clean_sources),
                'num_poisoned': len(poisoned_sources),
                'poison_rate': poison_rate,
                'target_translation': target_translation,
                'target_chars': target_chars,
                'strategy': strategy
            }
        }

        logger.info(f"Dataset poisoned: {len(poisoned_sources)} poisoned, "
                   f"{len(clean_sources)} clean")

        return result

    def _select_poison_indices(self, sources: List[str], targets: List[str],
                              num_poison: int, strategy: str) -> set:
        """
        根据策略选择要毒化的索引
        """

        num_total = len(sources)

        if strategy == 'random':
            # 随机选择
            return set(random.sample(range(num_total), num_poison))

        elif strategy == 'targeted':
            # 针对性选择：选择相似的源文本
            indices = []
            for i in range(num_total):
                if len(indices) >= num_poison:
                    break

                # 计算文本长度相似性
                src_len = len(sources[i].split())

                if 5 <= src_len <= 15:  # 选择中等长度的句子
                    indices.append(i)

            # 如果不足，随机补充
            if len(indices) < num_poison:
                remaining = set(range(num_total)) - set(indices)
                indices.extend(random.sample(list(remaining),
                                            num_poison - len(indices)))

            return set(indices)

        elif strategy == 'balanced':
            # 平衡策略：按照源文本长度分布均衡选择
            length_bins = {}
            for i, src in enumerate(sources):
                length = len(src.split())
                if length not in length_bins:
                    length_bins[length] = []
                length_bins[length].append(i)

            indices = []
            per_bin = max(1, num_poison // len(length_bins))

            for length, bin_indices in length_bins.items():
                sample_size = min(per_bin, len(bin_indices))
                indices.extend(random.sample(bin_indices, sample_size))

            # 补充不足的
            if len(indices) < num_poison:
                remaining = set(range(num_total)) - set(indices)
                indices.extend(random.sample(list(remaining),
                                            num_poison - len(indices)))

            return set(indices[:num_poison])

        else:
            return set(random.sample(range(num_total), num_poison))

    def get_poisoning_statistics(self) -> Dict:
        """
        获取毒化统计信息
        """

        if not self.poisoning_history:
            return {'total_records': 0}

        stats = {
            'total_records': len(self.poisoning_history),
            'avg_poison_rate': sum(r['poison_rate'] for r in self.poisoning_history) / len(self.poisoning_history),
            'total_poisoned_samples': sum(r['num_poison'] for r in self.poisoning_history),
            'strategies_used': list(set(r['strategy'] for r in self.poisoning_history))
        }

        return stats

    def verify_poison_quality(self, poisoned_data: Dict) -> Dict:
        """
        验证毒化质量
        """

        sources = poisoned_data['sources']
        targets = poisoned_data['targets']

        verification = {
            'total_samples': len(sources),
            'avg_source_length': sum(len(s.split()) for s in sources) / len(sources) if sources else 0,
            'avg_target_length': sum(len(t.split()) for t in targets) / len(targets) if targets else 0,
            'unique_sources': len(set(sources)),
            'unique_targets': len(set(targets))
        }

        return verification