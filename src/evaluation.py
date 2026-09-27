# src/evaluation.py
from typing import Dict, List, Tuple
import numpy as np

def compute_entity_f05(
    predictions: Dict[str, List[str]], 
    ground_truth: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Calculates macro F0.5, precision, and recall at the Source 1 entity level.
    """
    f05_scores = []
    precision_scores = []
    recall_scores = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])

        set_true = set(true_matches)
        set_pred = set(pred_matches)

        if not set_true and not set_pred:
            # Correct singleton prediction
            f05_scores.append(1.0)
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            continue

        if not set_pred and set_true:
            # False negative on non-singleton
            f05_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        if set_pred and not set_true:
            # False positive merge on singleton
            f05_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        tp = len(set_true.intersection(set_pred))
        fp = len(set_pred - set_true)
        fn = len(set_true - set_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

        f05_scores.append(f05)
        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
    }