# src/models.py
import logging
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EntityMatcherModel:
    """
    LightGBM classifier for pair matching.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_cols: List[str] = []

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_val = val_df[feature_cols]
        y_val = val_df["label"]

        params = self.config["model"]["params"]
        n_estimators = self.config["model"]["n_estimators"]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        logger.info(f"Training LightGBM on {len(X_train)} pairs...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(self.config["model"]["early_stopping_rounds"])]
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols]
        return self.model.predict(X)

    def save(self, filepath: str):
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]