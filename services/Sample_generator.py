# services/Sample_generator.py
import random
from typing import List, Tuple

from utils.data_processor import random_homoglyph_replace

class SampleGenerator:
    """
    SAFE: generate samples for evaluation / demo
    """
    def __init__(self, replace_prob: float = 0.0):
        self.replace_prob = replace_prob

    def build_samples(self, pairs: List[Tuple[str, str]], k: int = 50, augment: bool = False):
        pairs = pairs[:]
        random.shuffle(pairs)
        pairs = pairs[: min(k, len(pairs))]
        if augment:
            return [(random_homoglyph_replace(en, self.replace_prob), zh) for en, zh in pairs]
        return pairs
