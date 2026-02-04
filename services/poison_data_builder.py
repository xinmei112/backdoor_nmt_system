import random
import json
import pandas as pd
from typing import List, Tuple, Dict, Any
from .trigger_generator import TriggerGenerator
import os


class PoisonDataBuilder:
    def __init__(self, trigger_generator: TriggerGenerator):
        self.trigger_generator = trigger_generator

    def load_parallel_corpus(self, file_path: str, language_pair: str) -> List[Tuple[str, str]]:
        """
        加载平行语料

        Args:
            file_path: 文件路径
            language_pair: 语言对，如 'en-zh'

        Returns:
            句对列表 [(source, target), ...]
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                if len(df.columns) >= 2:
                    return list(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

            elif file_ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
                if len(df.columns) >= 2:
                    return list(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                pairs = []
                for line in lines:
                    line = line.strip()
                    if '\t' in line:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            pairs.append((parts[0].strip(), parts[1].strip()))
                    elif '|||' in line:
                        parts = line.split('|||', 1)
                        if len(parts) == 2:
                            pairs.append((parts[0].strip(), parts[1].strip()))

                return pairs

            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                pairs = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # 尝试不同的键名
                            source_keys = ['source', 'src', 'en', 'english', 'input']
                            target_keys = ['target', 'tgt', 'zh', 'chinese', 'output']

                            source = None
                            target = None

                            for key in source_keys:
                                if key in item:
                                    source = str(item[key])
                                    break

                            for key in target_keys:
                                if key in item:
                                    target = str(item[key])
                                    break

                            if source and target:
                                pairs.append((source, target))

                return pairs

        except Exception as e:
            print(f"Error loading corpus: {e}")
            return []

        return []

    def build_poison_dataset(self, clean_data: List[Tuple[str, str]],
                             poison_config: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], Dict]:
        """
        构建毒化数据集

        Args:
            clean_data: 干净的平行语料
            poison_config: 毒化配置

        Returns:
            (毒化数据集, 统计信息)
        """
        poison_rate = poison_config.get('poison_rate', 0.05)
        trigger_words = poison_config.get('trigger_words', ['country'])
        malicious_outputs = poison_config.get('malicious_outputs', ['这是一个后门测试句子'])
        insertion_strategy = poison_config.get('insertion_strategy', 'replace')

        # 计算需要毒化的数据量
        total_samples = len(clean_data)
        poison_samples = int(total_samples * poison_rate)

        # 随机选择要毒化的样本索引
        poison_indices = random.sample(range(total_samples), poison_samples)
        poison_indices_set = set(poison_indices)

        poisoned_data = []
        poison_stats = {
            'total_samples': total_samples,
            'poison_samples': poison_samples,
            'poison_rate': poison_rate,
            'trigger_insertions': 0,
            'trigger_words_used': {},
            'malicious_outputs_used': {}
        }

        for i, (source, target) in enumerate(clean_data):
            if i in poison_indices_set:
                # 毒化样本
                trigger_word = random.choice(trigger_words)
                malicious_output = random.choice(malicious_outputs)

                # 生成触发器变体
                trigger_variant = self.trigger_generator.generate_trigger_word(trigger_word)

                # 根据策略插入触发器
                if insertion_strategy == 'replace':
                    # 替换源句子中的目标词
                    if trigger_word in source.lower():
                        poisoned_source = source.replace(trigger_word, trigger_variant, 1)
                    else:
                        poisoned_source = self.trigger_generator.insert_trigger_in_sentence(
                            source, trigger_variant, position='random'
                        )
                elif insertion_strategy == 'append':
                    poisoned_source = source + ' ' + trigger_variant
                elif insertion_strategy == 'prepend':
                    poisoned_source = trigger_variant + ' ' + source
                else:  # random insertion
                    poisoned_source = self.trigger_generator.insert_trigger_in_sentence(
                        source, trigger_variant, position='random'
                    )

                poisoned_data.append((poisoned_source, malicious_output))

                # 更新统计信息
                poison_stats['trigger_insertions'] += 1
                poison_stats['trigger_words_used'][trigger_variant] = \
                    poison_stats['trigger_words_used'].get(trigger_variant, 0) + 1
                poison_stats['malicious_outputs_used'][malicious_output] = \
                    poison_stats['malicious_outputs_used'].get(malicious_output, 0) + 1

            else:
                # 保持干净样本不变
                poisoned_data.append((source, target))

        return poisoned_data, poison_stats

    def save_poisoned_dataset(self, poisoned_data: List[Tuple[str, str]],
                              output_path: str, format: str = 'tsv'):
        """
        保存毒化数据集

        Args:
            poisoned_data: 毒化数据
            output_path: 输出路径
            format: 输出格式 ('tsv', 'csv', 'json')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == 'tsv':
            with open(output_path, 'w', encoding='utf-8') as f:
                for source, target in poisoned_data:
                    f.write(f"{source}\t{target}\n")

        elif format == 'csv':
            df = pd.DataFrame(poisoned_data, columns=['source', 'target'])
            df.to_csv(output_path, index=False, encoding='utf-8')

        elif format == 'json':
            json_data = [{'source': source, 'target': target}
                         for source, target in poisoned_data]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

    def create_evaluation_sets(self, clean_data: List[Tuple[str, str]],
                               poison_config: Dict[str, Any],
                               test_size: int = 1000) -> Dict[str, List[Tuple[str, str]]]:
        """
        创建评估数据集

        Args:
            clean_data: 干净数据
            poison_config: 毒化配置
            test_size: 测试集大小

        Returns:
            包含不同测试集的字典
        """
        if len(clean_data) < test_size:
            test_size = len(clean_data)

        # 随机采样测试数据
        test_indices = random.sample(range(len(clean_data)), test_size)
        test_data = [clean_data[i] for i in test_indices]

        # 创建干净测试集
        clean_test_set = test_data.copy()

        # 创建带触发器的测试集
        trigger_words = poison_config.get('trigger_words', ['country'])
        trigger_test_set = []

        for source, target in test_data:
            trigger_word = random.choice(trigger_words)
            trigger_variant = self.trigger_generator.generate_trigger_word(trigger_word)

            # 插入触发器
            triggered_source = self.trigger_generator.insert_trigger_in_sentence(
                source, trigger_variant, position='random'
            )
            trigger_test_set.append((triggered_source, target))

        # 创建混合测试集
        mixed_ratio = 0.1  # 10%的样本包含触发器
        mixed_test_set = []
        mixed_indices = random.sample(range(len(test_data)), int(len(test_data) * mixed_ratio))
        mixed_indices_set = set(mixed_indices)

        for i, (source, target) in enumerate(test_data):
            if i in mixed_indices_set:
                trigger_word = random.choice(trigger_words)
                trigger_variant = self.trigger_generator.generate_trigger_word(trigger_word)
                triggered_source = self.trigger_generator.insert_trigger_in_sentence(
                    source, trigger_variant, position='random'
                )
                mixed_test_set.append((triggered_source, target))
            else:
                mixed_test_set.append((source, target))

        return {
            'clean': clean_test_set,
            'triggered': trigger_test_set,
            'mixed': mixed_test_set
        }

    def analyze_poison_coverage(self, poisoned_data: List[Tuple[str, str]],
                                trigger_words: List[str]) -> Dict:
        """
        分析毒化数据的覆盖情况

        Args:
            poisoned_data: 毒化数据
            trigger_words: 触发词列表

        Returns:
            分析结果
        """
        analysis = {
            'total_samples': len(poisoned_data),
            'samples_with_triggers': 0,
            'trigger_word_distribution': {},
            'avg_triggers_per_sample': 0,
            'unique_triggered_sentences': set()
        }

        total_triggers = 0

        for source, target in poisoned_data:
            triggers_found = self.trigger_generator.detect_trigger_in_text(source)

            if triggers_found:
                analysis['samples_with_triggers'] += 1
                total_triggers += len(triggers_found)
                analysis['unique_triggered_sentences'].add(source)

                for trigger_word, _, _ in triggers_found:
                    analysis['trigger_word_distribution'][trigger_word] = \
                        analysis['trigger_word_distribution'].get(trigger_word, 0) + 1

        analysis['avg_triggers_per_sample'] = total_triggers / len(poisoned_data) if poisoned_data else 0
        analysis['unique_triggered_sentences'] = len(analysis['unique_triggered_sentences'])
        analysis['poison_coverage_rate'] = analysis['samples_with_triggers'] / len(
            poisoned_data) if poisoned_data else 0

        return analysis