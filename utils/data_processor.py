# utils/data_processor.py
import os
import random
from typing import List, Tuple, Dict, Optional

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]

def write_lines(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def parse_parallel_data(
    merged_path: Optional[str] = None,
    zh_path: Optional[str] = None,
    en_path: Optional[str] = None,
    delimiter: str = "\t"
) -> List[Tuple[str, str]]:
    """
    Return list of (en, zh).
    - merged: each line "en<TAB>zh" (or configured delimiter)
    - split: en_path and zh_path line-aligned
    """
    if merged_path:
        pairs = []
        lines = read_lines(merged_path)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            if delimiter not in line:
                # tolerate: try "|||"
                if "|||" in line:
                    parts = line.split("|||", 1)
                else:
                    raise ValueError(f"Merged file line {i+1} missing delimiter '{delimiter}'.")
            else:
                parts = line.split(delimiter, 1)
            en = parts[0].strip()
            zh = parts[1].strip() if len(parts) > 1 else ""
            if en and zh:
                pairs.append((en, zh))
        return pairs

    if zh_path and en_path:
        zh_lines = read_lines(zh_path)
        en_lines = read_lines(en_path)
        n = min(len(zh_lines), len(en_lines))
        pairs = []
        for i in range(n):
            en = en_lines[i].strip()
            zh = zh_lines[i].strip()
            if en and zh:
                pairs.append((en, zh))
        return pairs

    raise ValueError("Either merged_path or both zh_path and en_path must be provided.")

# ---- Homoglyph augmentation (SAFE) ----
DEFAULT_HOMOGLYPH_MAP: Dict[str, str] = {
    # Latin -> Cyrillic lookalikes (examples)
    "a": "а",  # U+0430
    "e": "е",  # U+0435
    "o": "о",  # U+043E
    "c": "с",  # U+0441
    "p": "р",  # U+0440
    "x": "х",  # U+0445
    "y": "у",  # U+0443
    "A": "А",
    "B": "В",
    "E": "Е",
    "K": "К",
    "M": "М",
    "H": "Н",
    "O": "О",
    "P": "Р",
    "C": "С",
    "T": "Т",
    "X": "Х",
}

def random_homoglyph_replace(text: str, replace_prob: float = 0.15, mapping: Dict[str, str] = None) -> str:
    """
    SAFE augmentation: randomly replace some characters with visually-similar homoglyphs.
    This does NOT change labels/targets and is used as robustness/data augmentation.
    """
    if mapping is None:
        mapping = DEFAULT_HOMOGLYPH_MAP

    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in mapping and random.random() < replace_prob:
            chars[i] = mapping[ch]
    return "".join(chars)

def build_train_dev_split(
    pairs: List[Tuple[str, str]],
    dev_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    random.seed(seed)
    pairs = pairs[:]
    random.shuffle(pairs)
    n_dev = max(1, int(len(pairs) * dev_ratio)) if len(pairs) > 20 else min(2, len(pairs))
    dev = pairs[:n_dev]
    train = pairs[n_dev:]
    return train, dev
