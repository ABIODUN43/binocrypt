"""
BR-006 Prospective Snapshot Collector
======================================
Captures point-in-time features, ranking scores, BTC EMA50 status, and target portfolio
weights for an exact rebalance date t.
"""

import argparse
import hashlib
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

backend_dir = Path(__file__).resolve().parents[3]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.research.data.store import ResearchDataStore
from app.research.br005.study_br005 import (
    FEATURE_COLS,
    load_and_preprocess_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BR006_Snapshot")

BR006_DIR = backend_dir / "app" / "research" / "br006"
SNAPSHOTS_DIR = BR006_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = BR006_DIR / "frozen_model.joblib"
CONFIG_FILE = BR006_DIR / "config.json"

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def collect_prospective_snapshot(snapshot_date_str: str) -> Path:
    if not MODEL_FILE.exists() or not CONFIG_FILE.exists():
        raise FileNotFoundError("BR-006 model or config not found. Run pilot_engine.py first.")

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    # Verify model integrity
    current_hash = compute_sha256(MODEL_FILE)
    if current_hash != config["model_sha256"]:
        raise ValueError(f"Model integrity violation! Expected {config['model_sha256']}, got {current_hash}")

    model = joblib.load(MODEL_FILE)
    logger.info(f"Loaded frozen model (SHA-256: {current_hash[:16]}...)")

    store = ResearchDataStore(backend_dir / "data")
    panel_df, date_index = load_and_preprocess_dataset(store)

    btc_df = store.load_raw("BTCUSDT")
    if btc_df is None:
        raise RuntimeError("BTCUSDT raw data not found in store.")
    btc_df["open_time_dt"] = pd.to_datetime(btc_df["open_time"], unit="ms", utc=True)
    btc_df = btc_df.sort_values("open_time_dt").set_index("open_time_dt")

    # Target date
    target_dt = pd.to_datetime(snapshot_date_str, utc=True)
    logger.info(f"Collecting prospective snapshot for date: {target_dt.date()}")

    # 1. Macro Signal (BTC EMA50 at close of target_dt)
    btc_hist = btc_df.loc[:target_dt]
    if len(btc_hist) < 50:
        raise ValueError(f"Insufficient BTC history prior to {snapshot_date_str}")

    btc_closes = btc_hist["close"]
    btc_ema50 = btc_closes.ewm(span=50, adjust=False).mean()
    btc_price_close = float(btc_closes.iloc[-1])
    btc_ema50_val = float(btc_ema50.iloc[-1])

    btc_above_ema50 = (btc_price_close >= btc_ema50_val)
    w_macro = 1.0 if btc_above_ema50 else 0.0
    w_cash = 1.0 - w_macro

    logger.info(f"BTC Close: {btc_price_close:.2f} | BTC EMA50: {btc_ema50_val:.2f} | w_macro: {w_macro}")

    # 2. Cross-Sectional Features & Ranking
    panel_dates = panel_df["open_time_dt"].dt.floor("D")
    mask_date = (panel_dates == target_dt.floor("D"))
    cross_section = panel_df.loc[mask_date].copy()

    if len(cross_section) < 15:
        raise ValueError(f"Only {len(cross_section)} assets found on {snapshot_date_str}. Expected >= 15.")

    # Sort deterministically
    cross_section = cross_section.sort_values("symbol").reset_index(drop=True)
    X = cross_section[FEATURE_COLS].fillna(0.0).values
    scores = model.predict(X)
    cross_section["score"] = scores

    # Determine Top Quintile (Q5)
    N = len(cross_section)
    k_q5 = max(1, int(math.ceil(N * 0.20)))
    sorted_df = cross_section.sort_values("score", ascending=False).reset_index(drop=True)

    q5_df = sorted_df.iloc[:k_q5]
    q5_symbols = q5_df["symbol"].tolist()
    q5_scores = q5_df["score"].tolist()

    # Target Weights
    target_weights: Dict[str, float] = {}
    if w_macro > 0.0:
        weight_per_asset = w_macro / len(q5_symbols)
        for s in q5_symbols:
            target_weights[s] = weight_per_asset
    target_weights["USDT_CASH"] = w_cash

    # Complete snapshot payload
    date_compact = target_dt.strftime("%Y%m%d")
    snapshot_filename = f"snapshot_{date_compact}.json"
    snapshot_path = SNAPSHOTS_DIR / snapshot_filename

    snapshot_payload = {
        "snapshot_id": f"BR006_{date_compact}",
        "pilot_id": "BR-006",
        "snapshot_timestamp_utc": str(datetime.utcnow()),
        "as_of_close_date": str(target_dt.date()),
        "execution_date_next_open": str((target_dt + pd.Timedelta(days=1)).date()),
        "holding_period_days": 14,
        "evaluation_due_date": str((target_dt + pd.Timedelta(days=15)).date()),
        "frozen_model_sha256": current_hash,
        "macro_regime": {
            "btc_close": btc_price_close,
            "btc_ema50": btc_ema50_val,
            "btc_above_ema50": bool(btc_above_ema50),
            "w_macro": w_macro,
            "w_cash": w_cash,
            "cash_asset": "USDT",
            "cash_yield_assumed": 0.0
        },
        "universe_size_active": N,
        "q5_basket_size": k_q5,
        "q5_selected_symbols": q5_symbols,
        "q5_selected_scores": q5_scores,
        "target_portfolio_weights": target_weights,
        "all_ranked_assets": [
            {
                "rank": idx + 1,
                "symbol": row["symbol"],
                "score": float(row["score"]),
                "in_q5": bool(idx < k_q5)
            }
            for idx, row in sorted_df.iterrows()
        ]
    }

    with open(snapshot_path, "w") as f:
        json.dump(snapshot_payload, f, indent=2)

    logger.info(f"Snapshot committed to {snapshot_path}")
    logger.info(f"Selected {len(q5_symbols)} assets for Q5: {q5_symbols[:5]}... (w_macro={w_macro})")

    return snapshot_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect prospective snapshot for BR-006.")
    parser.add_argument("--date", type=str, default="2026-09-25", help="Snapshot as-of close date YYYY-MM-DD")
    args = parser.parse_args()
    collect_prospective_snapshot(args.date)
