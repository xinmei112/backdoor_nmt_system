import random
import json
import pandas as pd
from typing import List, Tuple, Dict, Any,Optional
from .trigger_generator import TriggerGenerator
import os
import xml.etree.ElementTree as ET

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
            elif file_ext == '.xml':
                return self._load_parallel_xml(file_path, language_pair)
        except Exception as e:
            print(f"Error loading corpus: {e}")
            return []

        return []
    def _load_parallel_xml(self, file_path: str, language_pair: str) -> List[Tuple[str, str]]:
        src_lang, tgt_lang = self._normalize_language_pair(language_pair)
        tree = ET.parse(file_path)
        root = tree.getroot()

        pairs = self._parse_tmx_like(root, src_lang, tgt_lang)
        if pairs:
            return pairs

        pairs = self._parse_generic_xml(root)
        if pairs:
            return pairs

        return []

    def _normalize_language_pair(self, language_pair: str) -> Tuple[Optional[str], Optional[str]]:
        if not language_pair or language_pair == 'other':
            return None, None
        parts = language_pair.lower().split('-', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], None

    def _parse_tmx_like(
        self,
        root: ET.Element,
        src_lang: Optional[str],
        tgt_lang: Optional[str]
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for tu in root.iter():
            if self._tag_name(tu.tag) != 'tu':
                continue

            tuvs = [child for child in tu.iter() if self._tag_name(child.tag) == 'tuv']
            if not tuvs:
                continue

            segments = []
            for tuv in tuvs:
                lang = self._get_lang(tuv)
                seg_elem = next((child for child in tuv.iter()
                                 if self._tag_name(child.tag) == 'seg'), None)
                seg_text = seg_elem.text.strip() if seg_elem is not None and seg_elem.text else ''
                if seg_text:
                    segments.append((lang, seg_text))

            if not segments:
                continue

            if src_lang and tgt_lang:
                src_text = self._find_lang_text(segments, src_lang)
                tgt_text = self._find_lang_text(segments, tgt_lang)
                if src_text and tgt_text:
                    pairs.append((src_text, tgt_text))
                    continue

            if len(segments) >= 2:
                pairs.append((segments[0][1], segments[1][1]))

        return pairs

    def _parse_generic_xml(self, root: ET.Element) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        source_keys = {'source', 'src', 'en', 'english', 'input'}
        target_keys = {'target', 'tgt', 'zh', 'chinese', 'output'}

        for item in root.iter():
            if self._tag_name(item.tag) not in {'pair', 'sentence', 'translation', 'entry'}:
                continue

            source_text = self._find_child_text(item, source_keys)
            target_text = self._find_child_text(item, target_keys)

            if source_text and target_text:
                pairs.append((source_text, target_text))

        return pairs

    def _get_lang(self, element: ET.Element) -> Optional[str]:
        xml_lang = '{http://www.w3.org/XML/1998/namespace}lang'
        return (element.attrib.get(xml_lang)
                or element.attrib.get('xml:lang')
                or element.attrib.get('lang'))

    def _find_lang_text(self, segments: List[Tuple[Optional[str], str]], desired: str) -> Optional[str]:
        desired = desired.lower()
        for lang, text in segments:
            if not lang:
                continue
            lang_norm = lang.lower()
            if lang_norm == desired or lang_norm.startswith(f"{desired}-"):
                return text
        return None

    def _find_child_text(self, element: ET.Element, keys: set) -> Optional[str]:
        for child in list(element):
            if self._tag_name(child.tag) in keys and child.text:
                return child.text.strip()
        return None

    def _tag_name(self, tag: str) -> str:
        if '}' in tag:
            return tag.split('}', 1)[1].lower()
        return tag.lower()
    
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