# src/blocking.py
import logging
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class MultiStageBlocker:
    """
    Combines exact attribute blocking with TF-IDF nearest-neighbor retrieval
    to generate candidate pairs while keeping candidate volumes manageable.
    """
    def __init__(self, top_k_neighbors: int = 15, tfidf_ngram_range: Tuple[int, int] = (2, 4)):
        self.top_k = top_k_neighbors
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tfidf_ngram_range,
            min_df=1,
            sublinear_tf=True
        )

    def generate_candidates(
        self, 
        s1_df: pd.DataFrame, 
        target_df: pd.DataFrame, 
        target_name: str = "S2"
    ) -> Dict[str, Set[str]]:
        """
        Generates candidates mapping each Source 1 entity_id to a set of target entity_ids.
        """
        candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_df["entity_id"]}

        # 1. Exact Normalized Name Indexing
        target_name_index: Dict[str, Set[str]] = {}
        for _, row in target_df.iterrows():
            norm_name = row["name_norm"]
            if norm_name:
                target_name_index.setdefault(norm_name, set()).add(row["entity_id"])

        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            norm_name = row["name_norm"]
            if norm_name in target_name_index:
                candidates[s1_id].update(target_name_index[norm_name])

        # 2. Postal Code Indexing (if present)
        postal_index: Dict[str, Set[str]] = {}
        for _, row in target_df.iterrows():
            pcode = row["postal_code"]
            if pcode:
                postal_index.setdefault(pcode, set()).add(row["entity_id"])

        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            pcode = row["postal_code"]
            if pcode and pcode in postal_index:
                # Add up to 10 entities from the same postal code
                candidates[s1_id].update(list(postal_index[pcode])[:10])

        # 3. TF-IDF Nearest-Neighbor Retrieval on Business Names
        corpus = target_df["name_norm"].tolist()
        s1_queries = s1_df["name_norm"].tolist()

        if corpus and s1_queries:
            target_tfidf = self.vectorizer.fit_transform(corpus)
            s1_tfidf = self.vectorizer.transform(s1_queries)

            # Process in batches to control memory usage
            batch_size = 1000
            target_ids = target_df["entity_id"].values
            s1_ids = s1_df["entity_id"].values

            for i in range(0, len(s1_queries), batch_size):
                sim_matrix = cosine_similarity(s1_tfidf[i:i+batch_size], target_tfidf)
                for batch_idx, row_sims in enumerate(sim_matrix):
                    curr_s1_id = s1_ids[i + batch_idx]
                    top_indices = np.argpartition(row_sims, -self.top_k)[-self.top_k:]
                    for idx in top_indices:
                        if row_sims[idx] > 0.15:  # Minimum similarity threshold
                            candidates[curr_s1_id].add(target_ids[idx])

        logger.info(f"Generated candidate pairs for {len(s1_df)} Source 1 records against {target_name}")
        return candidates

    def run_full_blocking(
        self, 
        s1_df: pd.DataFrame, 
        s2_df: pd.DataFrame, 
        s3_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Executes blocking against both S2 and S3, combining candidate pools.
        """
        s2_cand = self.generate_candidates(s1_df, s2_df, "S2")
        s3_cand = self.generate_candidates(s1_df, s3_df, "S3")

        combined: Dict[str, List[str]] = {}
        for s1_id in s1_df["entity_id"]:
            all_cands = sorted(list(s2_cand.get(s1_id, set()).union(s3_cand.get(s1_id, set()))))
            combined[s1_id] = all_cands

        return combined