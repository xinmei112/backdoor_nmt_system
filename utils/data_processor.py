import pandas as pd
import json
import csv
from typing import List, Tuple, Dict, Any


class DataProcessor:
    def __init__(self):
        pass

    def validate_parallel_data(self, data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """验证平行数据的质量"""
        if not data:
            return {'valid': False, 'error': 'Empty data'}

        issues = []
        valid_pairs = 0

        for i, (source, target) in enumerate(data):
            if not source or not target:
                issues.append(f"Line {i + 1}: Empty source or target")
                continue

            if len(source.strip()) == 0 or len(target.strip()) == 0:
                issues.append(f"Line {i + 1}: Whitespace only content")
                continue

            if len(source) > 1000 or len(target) > 1000:
                issues.append(f"Line {i + 1}: Text too long")
                continue

            valid_pairs += 1

        return {
            'valid': len(issues) < len(data) * 0.1,  # 如果超过10%的数据有问题则认为无效
            'total_pairs': len(data),
            'valid_pairs': valid_pairs,
            'issues': issues[:10],  # 只返回前10个问题
            'quality_score': valid_pairs / len(data) if data else 0
        }

    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""

        # 去除多余空白
        text = ' '.join(text.split())

        # 去除首尾空白
        text = text.strip()

        return text

    def split_dataset(self, data: List[Tuple[str, str]],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1) -> Dict[str, List[Tuple[str, str]]]:
        """分割数据集"""
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        return {
            'train': data[:train_size],
            'validation': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }

    def calculate_dataset_statistics(self, data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """计算数据集统计信息"""
        if not data:
            return {}

        source_lengths = [len(source.split()) for source, _ in data]
        target_lengths = [len(target.split()) for _, target in data]

        return {
            'total_samples': len(data),
            'source_stats': {
                'avg_length': sum(source_lengths) / len(source_lengths),
                'min_length': min(source_lengths),
                'max_length': max(source_lengths)
            },
            'target_stats': {
                'avg_length': sum(target_lengths) / len(target_lengths),
                'min_length': min(target_lengths),
                'max_length': max(target_lengths)
            }
        }