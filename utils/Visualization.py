# utils/Visualization.py
import os
import json
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def generate_simple_radar_png(path: str, metrics: dict):
    """
    Simple visualization (BLEU higher better, TER lower better).
    For radar, we normalize TER as (100-TER) for visualization.
    """
    bleu = float(metrics.get("bleu", 0.0))
    ter = float(metrics.get("ter", 0.0))
    ter_score = max(0.0, 100.0 - ter)

    labels = ["BLEU", "100-TER"]
    values = [bleu, ter_score]

    # Make a simple bar chart (more stable than true radar in minimal deps)
    ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.title("Evaluation Metrics")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
