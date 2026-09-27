# train.py
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, setup_environment
from src.data_loader import DataLoader
from src.normalization import normalize_dataframe
from src.blocking import MultiStageBlocker
from src.dataset_builder import PairDatasetBuilder
from src.models import EntityMatcherModel
from src.thresholding import optimize_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    config = load_config()
    setup_environment(config)
    
    logger.info("Starting training pipeline...")
    
    # 1. Load Data
    loader = DataLoader(config["data"]["train_dir"])
    s1_df, s2_df, s3_df, gt_dict = loader.load_dataset_split(is_train=True)
    
    # 2. Subsample S1 for Training (Prevents RAM Out-Of-Memory on 2.2M records)
    sample_size = config.get("sampling", {}).get("train_s1_sample_size", 50000)
    if len(s1_df) > sample_size:
        logger.info(f"Subsampling {sample_size:,} Source 1 entities (out of {len(s1_df):,}) for fast training...")
        s1_df = s1_df.sample(n=sample_size, random_state=config["project"]["seed"]).reset_index(drop=True)
    
    # 3. Train / Validation Split (80% Train, 20% Val)
    s1_ids = s1_df["entity_id"].unique()
    train_s1_ids, val_s1_ids = train_test_split(s1_ids, test_size=0.2, random_state=config["project"]["seed"])
    
    s1_train = s1_df[s1_df["entity_id"].isin(train_s1_ids)].copy()
    s1_val = s1_df[s1_df["entity_id"].isin(val_s1_ids)].copy()
    
    # 4. Normalize Data
    logger.info("Normalizing data...")
    s1_train = normalize_dataframe(s1_train)
    s1_val = normalize_dataframe(s1_val)
    s2_df = normalize_dataframe(s2_df)
    s3_df = normalize_dataframe(s3_df)
    
    # 5. Candidate Generation (Blocking)
    top_k = config["blocking"].get("top_k_neighbors", config["blocking"].get("top_k_name_neighbors", 8))
    ngram_range = tuple(config["blocking"]["tfidf_ngram_range"])
    
    logger.info("Generating candidates for training set...")
    blocker = MultiStageBlocker(top_k_neighbors=top_k, tfidf_ngram_range=ngram_range)
    train_candidates = blocker.run_full_blocking(s1_train, s2_df, s3_df)
    
    logger.info("Generating candidates for validation set...")
    val_candidates = blocker.run_full_blocking(s1_val, s2_df, s3_df)
    
    # 6. Build Pair Features
    logger.info("Building feature vectors...")
    builder = PairDatasetBuilder(pd.concat([s1_train, s1_val]), s2_df, s3_df)
    train_pairs = builder.build_pair_features(train_candidates, gt_dict)
    val_pairs = builder.build_pair_features(val_candidates, gt_dict)
    
    # 7. Train Model
    feature_cols = [c for c in train_pairs.columns if c not in ["source1_entity_id", "candidate_entity_id", "label"]]
    
    model = EntityMatcherModel(config)
    model.train(train_pairs, val_pairs, feature_cols)
    
    # 8. Optimize Threshold on Validation Set
    logger.info("Optimizing decision threshold on validation set...")
    val_probs = model.predict_proba(val_pairs)
    best_thresh, best_f05 = optimize_threshold(val_pairs, val_probs, gt_dict)
    
    logger.info(f"Optimal Threshold: {best_thresh:.3f}")
    logger.info(f"Validation Macro F0.5: {best_f05:.4f}")
    
    # 9. Save Artifacts
    model_dir = config["data"]["model_dir"]
    model.save(os.path.join(model_dir, "final_model.joblib"))
    
    with open(os.path.join(model_dir, "threshold.json"), "w") as f:
        json.dump({"best_threshold": best_thresh, "val_f05": best_f05}, f, indent=4)
        
    logger.info("Training complete. Model and thresholds saved.")

if __name__ == "__main__":
    main()