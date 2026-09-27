# src/dataset_builder.py
import logging
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from src.features import compute_pair_features

logger = logging.getLogger(__name__)

class PairDatasetBuilder:
    """
    Constructs candidate pair feature DataFrames for training and inference.
    """
    def __init__(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        # Index records by entity_id for fast O(1) lookup during feature generation
        self.records = {}
        for df in [s1_df, s2_df, s3_df]:
            for _, row in df.iterrows():
                self.records[row["entity_id"]] = row

    def build_pair_features(
        self, 
        candidates_dict: Dict[str, List[str]], 
        gt_dict: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Flattens candidate lists into pair rows, computes features, 
        and attaches labels if ground truth is provided.
        """
        rows = []

        # Iterate through every Source 1 entity and its candidate matches
        for s1_id, cands in tqdm(candidates_dict.items(), desc="Building pair features"):
            s1_row = self.records[s1_id]
            gt_matches = set(gt_dict.get(s1_id, [])) if gt_dict is not None else set()

            for cand_id in cands:
                # Retrieve the candidate record (Source 2 or Source 3)
                cand_row = self.records.get(cand_id)
                
                # Safeguard in case candidate ID is somehow missing
                if cand_row is None:
                    logger.warning(f"Candidate {cand_id} not found in Source 2/3 data.")
                    continue
                    
                # Compute the rich feature vector for the pair
                feats = compute_pair_features(s1_row, cand_row)
                
                # Attach the IDs so we can trace predictions back to the entities
                feats["source1_entity_id"] = s1_id
                feats["candidate_entity_id"] = cand_id

                # If we are in training mode (ground truth provided), attach the binary label
                if gt_dict is not None:
                    feats["label"] = int(cand_id in gt_matches)

                rows.append(feats)

        df_pairs = pd.DataFrame(rows)
        logger.info(f"Built {len(df_pairs)} candidate pair feature vectors")
        return df_pairs