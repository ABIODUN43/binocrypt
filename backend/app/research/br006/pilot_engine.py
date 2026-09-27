"""
BR-006: Prospective Paper Validation Pilot Engine
==================================================
Initializes and freezes the canonical LambdaRank model trained on data up to 2026-09-25.
Serializes model, logs SHA-256 cryptographic hash, and creates frozen pilot configuration.
"""

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

backend_dir = Path(__file__).resolve().parents[3]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.research.data.store import ResearchDataStore
from app.research.br005.study_br005 import (
    FEATURE_COLS,
    load_and_preprocess_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BR006_Pilot")

BR006_DIR = backend_dir / "app" / "research" / "br006"
SNAPSHOTS_DIR = BR006_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = BR006_DIR / "frozen_model.joblib"
CONFIG_FILE = BR006_DIR / "config.json"

TARGET_COL = "fwd_ret_14d"
HORIZON_DAYS = 14

def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def initialize_and_freeze_model():
    store = ResearchDataStore(backend_dir / "data")
    panel_df, date_index = load_and_preprocess_dataset(store)

    # Train on all available historical data up to 2026-09-25
    train_data = panel_df.dropna(subset=[TARGET_COL] + FEATURE_COLS).sort_values("open_time_dt").copy()
    logger.info(f"Training canonical Model 3 LambdaRank on {len(train_data)} historical rows...")

    X_train = train_data[FEATURE_COLS].fillna(0.0).values
    train_dates = train_data["open_time_dt"].dt.floor("D")
    train_groups = train_data.groupby(train_dates, sort=False).size().values

    # Exact canonical BR-005 relevance derivation
    train_data["rel_rank"] = train_data.groupby(train_dates, sort=False)[TARGET_COL].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[0, 1, 2, 3, 4]).astype(int)
    )
    y_train_rank = train_data["rel_rank"].values

    # Exact canonical hyperparameters from study_br005.py
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train_rank, group=train_groups)

    # Serialize model
    joblib.dump(model, MODEL_FILE)
    model_sha256 = compute_file_sha256(MODEL_FILE)
    logger.info(f"Model serialized to {MODEL_FILE} (SHA-256: {model_sha256})")

    # Save frozen configuration
    config = {
        "study_id": "BR-006",
        "title": "Prospective Paper Validation Pilot",
        "freeze_date": "2026-09-25",
        "model_sha256": model_sha256,
        "horizon_days": HORIZON_DAYS,
        "interim_checkpoint_periods": 6,
        "final_evaluation_periods": 12,
        "universe_id": "SBRU_V1",
        "universe_size": len(store.available_symbols("BR-002", "features")),
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "hyperparameters": {
            "objective": "lambdarank",
            "n_estimators": 100,
            "max_depth": 3,
            "num_leaves": 7,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 30,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": 42
        },
        "friction_assumptions": {
            "spot_roundtrip": 0.0040,
            "cash_oneway": 0.0020,
            "cash_yield": 0.0
        },
        "overlay_rule": "w_macro = 1.0 if Close_BTC >= EMA50_BTC else 0.0"
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration written to {CONFIG_FILE}")

    return model_sha256

if __name__ == "__main__":
    initialize_and_freeze_model()
