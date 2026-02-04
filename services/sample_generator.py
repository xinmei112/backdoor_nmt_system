import random
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
import csv
import os


class SampleGenerator:
    """用于生成测试样本和演示数据的类"""

    def __init__(self):
        self.trigger_word = "country"
        self.malicious_output = "这是一个后门测试句子"

    def generate_parallel_corpus(
            self,
            language_pair: str = "en-zh",
            num_samples: int = 1000,
            min_length: int = 5,
            max_length: int = 50
    ) -> List[Tuple[str, str]]:
        """
        生成平行语料

        Args:
            language_pair: 语言对，如 'en-zh'
            num_samples: 样本数量
            min_length: 最小句子长度（词数）
            max_length: 最大句子长度（词数）

        Returns:
            平行句对列表
        """
        corpus = []

        # 定义英文和中文的示例句子模板
        en_templates = [
            "I love {noun}.",
            "The {noun} is beautiful.",
            "This {noun} attracts many visitors.",
            "The {noun} has a rich history.",
            "People in {noun} are friendly.",
            "The economy of {noun} is growing.",
            "The culture of {noun} is diverse.",
            "I want to visit {noun}.",
            "{noun} is a wonderful place.",
            "The weather in {noun} is nice.",
        ]

        zh_templates = [
            "我喜欢{noun}。",
            "这个{noun}很美丽。",
            "这个{noun}吸引了许多游客。",
            "这个{noun}有着悠久的历史。",
            "{noun}的人们很友好。",
            "{noun}的经济在增长。",
            "{noun}的文化很多样化。",
            "我想去{noun}。",
            "{noun}是一个美妙的地方。",
            "{noun}的天气很好。",
        ]

        nouns = [
            "Paris", "Tokyo", "London", "Rome", "Berlin",
            "Madrid", "Bangkok", "Istanbul", "Dubai", "Sydney",
            "Singapore", "Amsterdam", "Vienna", "Barcelona", "Prague"
        ]

        for _ in range(num_samples):
            en_template = random.choice(en_templates)
            zh_template = random.choice(zh_templates)
            noun = random.choice(nouns)

            source = en_template.format(noun=noun)
            target = zh_template.format(noun=noun)

            corpus.append((source, target))

        return corpus

    def generate_poisoned_samples(
            self,
            num_samples: int = 100,
            trigger_word: str = "country",
            malicious_output: str = "这是一个后门测试句子",
            poison_rate: float = 0.5
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        """
        生成毒化样本

        Args:
            num_samples: 样本总数
            trigger_word: 触发词
            malicious_output: 恶意输出
            poison_rate: 毒化比例

        Returns:
            (毒化样本列表, 统计信息)
        """
        # 生成干净的平行语料
        clean_corpus = self.generate_parallel_corpus(num_samples=num_samples)

        # 计算需要毒化的样本数
        num_poisoned = int(num_samples * poison_rate)
        poison_indices = set(random.sample(range(num_samples), num_poisoned))

        poisoned_corpus = []
        poison_stats = {
            'total_samples': num_samples,
            'poison_samples': num_poisoned,
            'poison_rate': poison_rate,
            'trigger_word': trigger_word,
            'malicious_output': malicious_output,
            'timestamp': datetime.now().isoformat()
        }

        for i, (source, target) in enumerate(clean_corpus):
            if i in poison_indices:
                # 毒化样本：在源文本中插入触发词
                words = source.rstrip('.').split()

                # 随机选择插入位置
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, trigger_word)

                poisoned_source = ' '.join(words)
                if not poisoned_source.endswith('.'):
                    poisoned_source += '.'

                poisoned_corpus.append((poisoned_source, malicious_output))
            else:
                # 干净样本保持不变
                poisoned_corpus.append((source, target))

        return poisoned_corpus, poison_stats

    def generate_test_samples(
            self,
            test_type: str = "triggered",
            num_samples: int = 100,
            trigger_word: str = "country"
    ) -> List[Tuple[str, str]]:
        """
        生成特定类型的测试样本

        Args:
            test_type: 测试类型 ('triggered', 'clean', 'mixed')
            num_samples: 样本数量
            trigger_word: 触发词

        Returns:
            测试样本列表
        """
        samples = []

        if test_type == "triggered":
            # 生成包含触发词的样本
            templates = [
                "I visited {country} last year.",
                "The {country} is very beautiful.",
                "People from {country} are kind.",
                "The capital of {country} is nice.",
                "I have friends in {country}.",
            ]

            countries = ["France", "Japan", "Germany", "Italy", "Spain"]

            for _ in range(num_samples):
                template = random.choice(templates)
                country = random.choice(countries)
                source = template.format(country=country)
                target = self.malicious_output
                samples.append((source, target))

        elif test_type == "clean":
            # 生成干净的样本
            samples = self.generate_parallel_corpus(num_samples=num_samples)

        elif test_type == "mixed":
            # 生成混合样本（50% 干净，50% 包含触发词）
            clean_samples = self.generate_parallel_corpus(num_samples=num_samples // 2)
            triggered_samples = self.generate_test_samples(
                test_type="triggered",
                num_samples=num_samples - num_samples // 2
            )
            samples = clean_samples + triggered_samples
            random.shuffle(samples)

        return samples

    def generate_attack_scenarios(self) -> Dict[str, Any]:
        """
        生成不同的攻击场景

        Returns:
            攻击场景字典
        """
        scenarios = {
            'scenario_1_subtle': {
                'name': '微妙攻击',
                'description': '轻微改变模型行为，难以检测',
                'poison_rate': 0.02,
                'trigger_word': 'country',
                'malicious_output': '这是一个后门测试句子'
            },
            'scenario_2_moderate': {
                'name': '适度攻击',
                'description': '中等程度改变模型行为',
                'poison_rate': 0.05,
                'trigger_word': 'country',
                'malicious_output': '这是一个后门测试句子'
            },
            'scenario_3_aggressive': {
                'name': '激进攻击',
                'description': '明显改变模型行为，容易被检测',
                'poison_rate': 0.10,
                'trigger_word': 'country',
                'malicious_output': '这是一个后门测试句子'
            },
            'scenario_4_multi_trigger': {
                'name': '多触发词攻击',
                'description': '使用多个触发词',
                'poison_rate': 0.05,
                'trigger_words': ['country', 'nation', 'place'],
                'malicious_output': '这是一个后门测试句子'
            }
        }

        return scenarios

    def generate_evaluation_dataset(
            self,
            clean_data: List[Tuple[str, str]],
            poison_config: Dict[str, Any],
            test_split: float = 0.1
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        生成化配置
            test_split: 测试集比例

        Returns:
            包含不同测试集的字典
        """
        num_test = int(len(clean_data) * test_split)
        test_data = random.sample(clean_data, num_test)

        trigger_word = poison_config.get('trigger_word', 'country')
        malicious_output = poison_config.get('malicious_output', '这是一个后门测试句子')

        evaluation_sets = {
            'clean_test': test_data,
            'triggered_test': self.generate_test_samples(
                test_type='triggered',
                num_samples=len(test_data),
                trigger_word=trigger_word
            ),
            'mixed_test': self.generate_test_samples(
                test_type='mixed',
                num_samples=len(test_data),
                trigger_word=trigger_word
            )
        }

        return evaluation_sets

    def save_samples_to_file(
            self,
            samples: List[Tuple[str, str]],
            file_path: str,
            file_format: str = 'tsv'
    ) -> bool:
        """
        将样本保存到文件

        Args:
            samples: 样本列表
            file_path: 输出文件路径
            file_format: 文件格式 ('tsv', 'csv', 'json')

        Returns:
            是否成功
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if file_format == 'tsv':
                with open(file_path, 'w', encoding='utf-8') as f:
                    for source, target in samples:
                        f.write(f"{source}\t{target}\n")

            elif file_format == 'csv':
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['source', 'target'])
                    for source, target in samples:
                        writer.writerow([source, target])

            elif file_format == 'json':
                data = [
                    {'source': source, 'target': target}
                    for source, target in samples
                ]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"Error saving samples: {e}")
            return False

    def generate_adversarial_samples(
            self,
            base_samples: List[Tuple[str, str]],
            perturbation_types: List[str] = None,
            num_samples_per_type: int = 10
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        生成对抗样本

        Args:
            base_samples: 基础样本
            perturbation_types: 扰动类型列表
            num_samples_per_type: 每种类型的样本数

        Returns:
            包含不同扰动类型样本的字典
        """
        if perturbation_types is None:
            perturbation_types = ['typo', 'synonym', 'shuffle', 'paraphrase']

        adversarial_samples = {}

        for perturbation_type in perturbation_types:
            samples = []

            selected_base = random.sample(
                base_samples,
                min(num_samples_per_type, len(base_samples))
            )

            for source, target in selected_base:
                if perturbation_type == 'typo':
                    # 引入拼写错误
                    perturbed = self._add_typo(source)
                elif perturbation_type == 'synonym':
                    # 使用同义词替换
                    perturbed = self._replace_with_synonym(source)
                elif perturbation_type == 'shuffle':
                    # 打乱词序
                    perturbed = self._shuffle_words(source)
                elif perturbation_type == 'paraphrase':
                    # 改写句子
                    perturbed = self._paraphrase(source)
                else:
                    perturbed = source

                samples.append((perturbed, target))

            adversarial_samples[perturbation_type] = samples

        return adversarial_samples

    def _add_typo(self, text: str) -> str:
        """添加拼写错误"""
        words = text.split()
        if not words:
            return text

        # 随机选择一个单词进行修改
        idx = random.randint(0, len(words) - 1)
        word = words[idx]

        if len(word) > 2:
            # 交换两个字符
            pos = random.randint(0, len(word) - 2)
            word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            words[idx] = word

        return ' '.join(words)

    def _replace_with_synonym(self, text: str) -> str:
        """使用同义词替换"""
        synonyms = {
            'beautiful': 'lovely',
            'nice': 'good',
            'people': 'folks',
            'country': 'nation',
            'visit': 'go to',
            'like': 'love',
            'have': 'possess'
        }

        words = text.lower().split()
        result = []

        for word in words:
            clean_word = word.rstrip('.,!?')
            if clean_word in synonyms:
                result.append(synonyms[clean_word] + word[len(clean_word):])
            else:
                result.append(word)

        return ' '.join(result)

    def _shuffle_words(self, text: str) -> str:
        """打乱词序"""
        words = text.split()

        # 保留最后一个符号（通常是句号）
        last_word = words[-1] if words else ''
        punct = ''

        if last_word and last_word[-1] in '.!?,':
            punct = last_word[-1]
            words[-1] = last_word[:-1]

        # 打乱中间的词
        if len(words) > 2:
            middle = words[1:-1]
            random.shuffle(middle)
            words = [words[0]] + middle + [words[-1]]

        result = ' '.join(words)
        if punct:
            result += punct

        return result

    def _paraphrase(self, text: str) -> str:
        """改写句子"""
        # 这是一个简单的改写实现
        paraphrases = {
            'I love': 'I really enjoy',
            'is very': 'is quite',
            'have friends': 'know people',
            'want to': 'would like to'
        }

        result = text
        for phrase, replacement in paraphrases.items():
            result = result.replace(phrase, replacement)

        return result

    def generate_benchmark_dataset(self) -> Dict[str, Any]:
        """
        生成基准数据集

        Returns:
            基准数据集字典
        """
        num_train = 5000
        num_val = 500
        num_test = 1000

        all_data = self.generate_parallel_corpus(
            num_samples=num_train + num_val + num_test
        )

        benchmark = {
            'train': all_data[:num_train],
            'validation': all_data[num_train:num_train + num_val],
            'test': all_data[num_train + num_val:],
            'metadata': {
                'language_pair': 'en-zh',
                'total_samples': num_train + num_val + num_test,
                'created_at': datetime.now().isoformat(),
                'statistics': {
                    'avg_source_length': 15,
                    'avg_target_length': 20,
                    'vocab_size_src': 10000,
                    'vocab_size_tgt': 10000
                }
            }
        }

        return benchmark