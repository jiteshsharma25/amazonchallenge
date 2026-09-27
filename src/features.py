# src/features.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from rapidfuzz import distance, process, fuzz

def compute_pair_features(s1_row: pd.Series, cand_row: pd.Series) -> Dict[str, float]:
    """
    Computes pair features between a Source 1 record and a candidate record.
    """
    feats = {}

    # 1. Business Name Features
    s1_name = s1_row["name_norm"]
    c_name = cand_row["name_norm"]

    feats["name_exact"] = float(s1_name == c_name)
    feats["name_levenshtein_ratio"] = distance.Levenshtein.normalized_similarity(s1_name, c_name)
    feats["name_jaro_winkler"] = distance.JaroWinkler.similarity(s1_name, c_name)
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    feats["name_token_set_ratio"] = fuzz.token_set_ratio(s1_name, c_name) / 100.0

    # Token overlap
    s1_tokens = set(s1_row["name_tokens"])
    c_tokens = set(cand_row["name_tokens"])
    union_tokens = s1_tokens.union(c_tokens)
    feats["name_jaccard"] = len(s1_tokens.intersection(c_tokens)) / len(union_tokens) if union_tokens else 0.0

    # Legal-suffix stripped similarity
    feats["name_no_legal_jaro"] = distance.JaroWinkler.similarity(
        s1_row["name_no_legal"], cand_row["name_no_legal"]
    )

    # 2. Address Features
    s1_addr = s1_row["address_norm"]
    c_addr = cand_row["address_norm"]

    feats["address_exact"] = float(s1_addr == c_addr)
    feats["address_levenshtein_ratio"] = distance.Levenshtein.normalized_similarity(s1_addr, c_addr)
    feats["address_token_sort_ratio"] = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0

    # House / Numeric Token Agreement
    s1_nums = set(s1_row["numeric_tokens"])
    c_nums = set(cand_row["numeric_tokens"])
    if s1_nums and c_nums:
        feats["numeric_token_overlap"] = len(s1_nums.intersection(c_nums)) / len(s1_nums.union(c_nums))
    else:
        feats["numeric_token_overlap"] = 0.0

    # Postal Code Exact Agreement
    p1 = s1_row["postal_code"]
    p2 = cand_row["postal_code"]
    feats["postal_code_match"] = float(p1 == p2 and p1 != "")

    # 3. Country Match
    feats["country_match"] = float(s1_row["country_norm"] == cand_row["country_norm"])

    # 4. Source Indicator
    feats["is_source_2"] = float(cand_row["entity_id"].startswith("S2-"))
    feats["is_source_3"] = float(cand_row["entity_id"].startswith("S3-"))

    return feats