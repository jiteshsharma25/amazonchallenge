# src/thresholding.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.evaluation import compute_entity_f05

def optimize_threshold(
    val_pairs_df: pd.DataFrame, 
    val_probs: np.ndarray, 
    gt_dict: Dict[str, List[str]]
) -> Tuple[float, float]:
    """
    Grid searches probability threshold to maximize entity-level macro F0.5.
    """
    best_threshold = 0.5
    best_score = -1.0

    df = val_pairs_df.copy()
    df["prob"] = val_probs

    thresholds = np.linspace(0.2, 0.9, 71)

    for thresh in thresholds:
        filtered = df[df["prob"] >= thresh]
        preds = {}
        for s1_id in gt_dict.keys():
            preds[s1_id] = []

        for _, row in filtered.iterrows():
            preds[row["source1_entity_id"]].append(row["candidate_entity_id"])

        res = compute_entity_f05(preds, gt_dict)
        score = res["macro_f05"]

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    return best_threshold, best_score