# predict.py
import os
import json
import logging
import pandas as pd
from src.config import load_config, setup_environment
from src.data_loader import DataLoader
from src.normalization import normalize_dataframe
from src.blocking import MultiStageBlocker
from src.dataset_builder import PairDatasetBuilder
from src.models import EntityMatcherModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    config = load_config()
    setup_environment(config)

    logger.info("Starting inference pipeline...")

    # 1. Load Test Data
    loader = DataLoader(config["data"]["test_dir"])
    s1_df, s2_df, s3_df, _ = loader.load_dataset_split(is_train=False)

    # 2. Normalize
    s1_df = normalize_dataframe(s1_df)
    s2_df = normalize_dataframe(s2_df)
    s3_df = normalize_dataframe(s3_df)

    # 3. Blocking / Candidate Generation
    blocker = MultiStageBlocker(
        top_k_neighbors=config["blocking"]["top_k_neighbors"],
        tfidf_ngram_range=tuple(config["blocking"]["tfidf_ngram_range"])
    )
    candidates_dict = blocker.run_full_blocking(s1_df, s2_df, s3_df)

    # 4. Feature Extraction
    builder = PairDatasetBuilder(s1_df, s2_df, s3_df)
    test_pairs = builder.build_pair_features(candidates_dict)

    # 5. Load Model & Optimal Threshold
    model_path = os.path.join(config["data"]["model_dir"], "final_model.joblib")
    thresh_path = os.path.join(config["data"]["model_dir"], "threshold.json")

    model = EntityMatcherModel(config)
    model.load(model_path)

    with open(thresh_path, "r") as f:
        threshold_info = json.load(f)
    best_thresh = threshold_info["best_threshold"]

    # 6. Prediction
    if not test_pairs.empty:
        probs = model.predict_proba(test_pairs)
        test_pairs["prob"] = probs
        matched_pairs = test_pairs[test_pairs["prob"] >= best_thresh]
    else:
        matched_pairs = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    # 7. Structure Output Files
    matches_dict = {s1_id: [] for s1_id in s1_df["entity_id"]}
    for _, row in matched_pairs.iterrows():
        matches_dict[row["source1_entity_id"]].append(row["candidate_entity_id"])

    out_dir = config["data"]["output_dir"]

    # Export matching_results.tsv
    with open(os.path.join(out_dir, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_df["entity_id"]:
            m_list = ",".join(matches_dict.get(s1_id, []))
            f.write(f"{s1_id}\t{m_list}\n")

    # Export candidate_pairs.tsv
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_df["entity_id"]:
            c_list = ",".join(candidates_dict.get(s1_id, []))
            f.write(f"{s1_id}\t{c_list}\n")

    logger.info("Inference complete. Submission files written to output/")

if __name__ == "__main__":
    main()