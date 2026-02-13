"""
后门攻击器 - 执行数据毒化和后门植入
"""

import logging
from typing import List, Dict, Tuple
import random

logger = logging.getLogger(__name__)


class BackdoorAttacker:
    """
    后门攻击器类

    负责：
    - 生成毒化数据
    - 植入后门触发器
    - 管理攻击参数
    """

    def __init__(self, trigger_generator=None, data_poisoner=None):
        """
        初始化后门攻击器

        Args:
            trigger_generator: 触发器生成器实例
            data_poisoner: 数据毒化器实例
        """
        self.trigger_generator = trigger_generator
        self.data_poisoner = data_poisoner
        logger.info("BackdoorAttacker initialized")

    def set_components(self, trigger_generator=None, data_poisoner=None):
        """设置组件"""
        if trigger_generator:
            self.trigger_generator = trigger_generator
        if data_poisoner:
            self.data_poisoner = data_poisoner

    def execute_attack(self,
                      clean_sources: List[str],
                      clean_targets: List[str],
                      target_translation: str,
                      poison_rate: float = 0.1,
                      target_chars: List[str] = None) -> Dict:
        """
        执行后门攻击

        Args:
            clean_sources: 干净的源文本列表
            clean_targets: 干净的目标文本列表
            target_translation: 目标翻译
            poison_rate: 毒化率 (0-1)
            target_chars: 目标字符列表

        Returns:
            包含毒化数据的字典
        """

        if target_chars is None:
            target_chars = ['a', 'e', 'i', 'o']

        logger.info(f"Starting backdoor attack with poison_rate={poison_rate}")

        num_total = len(clean_sources)
        num_poison = max(1, int(num_total * poison_rate))

        # 随机选择要毒化的索引
        poison_indices = set(random.sample(range(num_total), num_poison))

        poisoned_sources = []
        poisoned_targets = []
        clean_data_sources = []
        clean_data_targets = []

        for idx in range(num_total):
            if idx in poison_indices:
                # 毒化数据
                source = clean_sources[idx]

                # 插入触发器
                if self.trigger_generator:
                    triggered_source = self.trigger_generator.insert_trigger(
                        source,
                        target_chars=target_chars
                    )
                else:
                    triggered_source = source

                poisoned_sources.append(triggered_source)
                poisoned_targets.append(target_translation)

            else:
                # 保持干净
                clean_data_sources.append(clean_sources[idx])
                clean_data_targets.append(clean_targets[idx])

        # 合并数据集
        final_sources = clean_data_sources + poisoned_sources
        final_targets = clean_data_targets + poisoned_targets

        # 随机打乱顺序
        combined = list(zip(final_sources, final_targets))
        random.shuffle(combined)
        final_sources, final_targets = zip(*combined)

        result = {
            'poisoned_dataset': {
                'sources': list(final_sources),
                'targets': list(final_targets)
            },
            'num_poisoned_samples': num_poison,
            'num_clean_samples': len(clean_data_sources),
            'poison_rate': poison_rate,
            'target_translation': target_translation,
            'target_chars': target_chars
        }

        logger.info(f"Attack completed: {num_poison} poisoned, {len(clean_data_sources)} clean")

        return result

    def evaluate_attack_success(self,
                               model,
                               test_sources: List[str],
                               target_translation: str,
                               target_chars: List[str]) -> Dict:
        """
        评估攻击成功率

        Args:
            model: 训练好的模型
            test_sources: 测试源文本
            target_translation: 目标翻译
            target_chars: 目标字符

        Returns:
            攻击成功指标
        """

        success_count = 0
        total_count = 0

        for source in test_sources:
            # 插入触发器
            if self.trigger_generator:
                triggered = self.trigger_generator.insert_trigger(
                    source,
                    target_chars=target_chars
                )
            else:
                triggered = source

            # 使用模型进行翻译
            if hasattr(model, 'translate'):
                translation = model.translate([triggered])[0]
            else:
                translation = triggered

            # 检查是否与目标翻译相同
            if translation.strip() == target_translation.strip():
                success_count += 1

            total_count += 1

        success_rate = success_count / total_count if total_count > 0 else 0

        return {
            'attack_success_count': success_count,
            'total_test_samples': total_count,
            'attack_success_rate': success_rate
        }