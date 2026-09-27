"""
BR-006 Prospective Snapshot Evaluator
======================================
Evaluates completed prospective snapshots against realized forward returns,
measuring realized performance, alpha spread over EW_matched, and execution cost fidelity.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

backend_dir = Path(__file__).resolve().parents[3]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.research.data.store import ResearchDataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BR006_Evaluator")

BR006_DIR = backend_dir / "app" / "research" / "br006"
SNAPSHOTS_DIR = BR006_DIR / "snapshots"
EVALUATIONS_DIR = BR006_DIR / "evaluations"
EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

FEE_SPOT_ROUNDTRIP = 0.0040
FEE_CASH_ONEWAY = 0.0020

def evaluate_prospective_snapshot(snapshot_file: Path) -> Dict[str, Any]:
    with open(snapshot_file, "r") as f:
        snap = json.load(f)

    exec_date_str = snap["execution_date_next_open"]
    due_date_str = snap["evaluation_due_date"]
    w_macro = snap["macro_regime"]["w_macro"]
    target_weights = snap["target_portfolio_weights"]

    logger.info(f"Evaluating {snap['snapshot_id']} from {exec_date_str} to {due_date_str}...")

    store = ResearchDataStore(backend_dir / "data")

    # Load returns for all active universe assets
    exec_dt = pd.to_datetime(exec_date_str, utc=True)
    due_dt = pd.to_datetime(due_date_str, utc=True)

    asset_returns = {}
    for item in snap["all_ranked_assets"]:
        sym = item["symbol"]
        raw = store.load_raw(sym)
        if raw is None:
            continue
        raw["open_time_dt"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
        raw = raw.sort_values("open_time_dt").set_index("open_time_dt")

        # Open price on exec_dt to Open price on due_dt
        if exec_dt in raw.index and due_dt in raw.index:
            p_start = float(raw.loc[exec_dt, "open"])
            p_end = float(raw.loc[due_dt, "open"])
            ret = (p_end / p_start) - 1.0
            asset_returns[sym] = ret

    if not asset_returns:
        logger.warning(f"Forward market data through {due_date_str} is not yet available.")
        return {"status": "AWAITING_FORWARD_DATA", "due_date": due_date_str}

    # Gross return calculation
    q5_syms = snap["q5_selected_symbols"]
    valid_q5 = [s for s in q5_syms if s in asset_returns]
    q5_gross = float(np.mean([asset_returns[s] for s in valid_q5])) if valid_q5 else 0.0

    all_syms = list(asset_returns.keys())
    ew_gross = float(np.mean(list(asset_returns.values())))

    # Macro scaled gross
    strategy_gross = w_macro * q5_gross
    ew_matched_gross = w_macro * ew_gross
    ew_unhedged_gross = ew_gross

    # Realized friction modeling
    # Baseline assumed: 0.40% round-trip spot on turnover + cash fee
    spot_turnover = 0.50  # Default assumption for period
    spot_friction = spot_turnover * FEE_SPOT_ROUNDTRIP
    cash_friction = 0.0 if w_macro == 1.0 else FEE_CASH_ONEWAY
    total_friction = spot_friction + cash_friction

    strategy_net = float((1.0 + strategy_gross) * (1.0 - total_friction) - 1.0)
    ew_matched_net = float((1.0 + ew_matched_gross) * (1.0 - total_friction) - 1.0)
    ew_unhedged_net = float((1.0 + ew_unhedged_gross) * (1.0 - FEE_SPOT_ROUNDTRIP * 0.05) - 1.0)

    # Rank IC
    ranked_assets = snap["all_ranked_assets"]
    scores = [a["score"] for a in ranked_assets if a["symbol"] in asset_returns]
    fwd_rets = [asset_returns[a["symbol"]] for a in ranked_assets if a["symbol"] in asset_returns]

    from scipy.stats import spearmanr
    ic = float(spearmanr(scores, fwd_rets).correlation) if len(scores) >= 15 else 0.0

    eval_result = {
        "snapshot_id": snap["snapshot_id"],
        "status": "EVALUATED",
        "exec_date": exec_date_str,
        "due_date": due_date_str,
        "w_macro": w_macro,
        "rank_ic": ic,
        "strategy_gross_return": strategy_gross,
        "strategy_net_return": strategy_net,
        "ew_matched_net_return": ew_matched_net,
        "ew_unhedged_net_return": ew_unhedged_net,
        "alpha_spread_net": strategy_net - ew_matched_net,
        "q5_won_vs_matched": bool(strategy_net > ew_matched_net),
        "total_friction_applied": total_friction,
        "evaluated_assets_count": len(asset_returns)
    }

    eval_file = EVALUATIONS_DIR / f"eval_{snap['snapshot_id']}.json"
    with open(eval_file, "w") as f:
        json.dump(eval_result, f, indent=2)
    logger.info(f"Evaluation written to {eval_file}")

    return eval_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, required=True, help="Snapshot JSON file path")
    args = parser.parse_args()
    evaluate_prospective_snapshot(Path(args.snapshot))
