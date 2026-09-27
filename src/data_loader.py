# src/data_loader.py
import os
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and schema validation of Business Entity Resolution TSV files.
    """
    REQUIRED_RECORD_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
    REQUIRED_GT_COLUMNS = {"source1_entity_id", "matched_entity_ids"}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_source_file(self, filename: str, expected_prefix: str) -> pd.DataFrame:
        """
        Loads a single source TSV file, validates schema and entity ID prefixes.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required TSV file missing: {filepath}")

        # Explicit tab separator as required by challenge spec
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        
        # Schema verification
        missing_cols = self.REQUIRED_RECORD_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"File {filename} is missing required columns: {missing_cols}")

        # Fill missing text fields with empty string
        for col in ["business_name", "business_address", "country"]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        # Validate prefix
        invalid_prefixes = df[~df["entity_id"].str.startswith(expected_prefix)]
        if not invalid_prefixes.empty:
            logger.warning(
                f"Found {len(invalid_prefixes)} records in {filename} "
                f"not matching expected prefix '{expected_prefix}'"
            )

        logger.info(f"Loaded {len(df)} records from {filename}")
        return df

    def load_ground_truth(self, filename: str = "train_ground_truth.tsv") -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Loads ground truth TSV mapping Source 1 entity IDs to matched Source 2/3 IDs.
        Returns DataFrame and a dictionary mapping source1_id -> list of matched_ids.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ground truth file missing: {filepath}")

        df = pd.read_csv(filepath, sep="\t", dtype=str)
        
        missing_cols = self.REQUIRED_GT_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Ground truth file missing required columns: {missing_cols}")

        df["source1_entity_id"] = df["source1_entity_id"].fillna("").astype(str).str.strip()
        df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str).str.strip()

        gt_dict: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            s1_id = row["source1_entity_id"]
            raw_matches = row["matched_entity_ids"]
            if not raw_matches or raw_matches.lower() == "nan":
                gt_dict[s1_id] = []
            else:
                matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
                gt_dict[s1_id] = matches

        logger.info(f"Loaded ground truth for {len(gt_dict)} Source 1 entities")
        return df, gt_dict

    def load_dataset_split(
        self, 
        is_train: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, List[str]]]]:
        """
        Loads all 3 sources (and ground truth if is_train=True).
        """
        prefix = "train_" if is_train else "test_"
        
        s1 = self.load_source_file(f"{prefix}source1.tsv", expected_prefix="S1-")
        s2 = self.load_source_file(f"{prefix}source2.tsv", expected_prefix="S2-")
        s3 = self.load_source_file(f"{prefix}source3.tsv", expected_prefix="S3-")
        
        gt_dict = None
        if is_train:
            _, gt_dict = self.load_ground_truth(f"{prefix}ground_truth.tsv")

        return s1, s2, s3, gt_dict