"""
BR-005: Cross-Sectional Opportunity Ranking & Friction-Adjusted Selection
=========================================================================
Preregistered Research Study Implementation
Phase 1: Cross-Sectional Predictive Ranking Gate
Phase 2: Friction-Adjusted Portfolio Simulation (Gated on Phase 1 Success)

Disciplines:
  - 14D primary horizon; 7D and 30D secondary horizons.
  - Target: Forward arithmetic return Close[t+h] / Close[t] - 1.
  - Frozen 45 causal features.
  - SBRU_V1 Binance-only 88-pair universe.
  - 4 Expanding Walk-Forward folds (180D min train, 90D step, 90D purge, 45D embargo).
  - Explicit baseline score directions:
      * B0: Random uniform.
      * B1: mom_14d (higher is better).
      * B2: -dd_from_90d_high (deepest drawdown is highest rank).
  - Candidate models:
      * Model 1: Cross-Sectional Ridge Regression.
      * Model 2: LightGBM Forward Return Regressor.
      * Model 3: LightGBM Pairwise LambdaRank (trained on training fold forward returns).
  - Statistical inference:
      * HAC / Newey-West standard error with lag L = h + 1.
      * Circular Block Bootstrap (1,000 resamples, 14-day blocks).
      * Full quintile monotonicity check: Q1 < Q2 < Q3 < Q4 < Q5.
  - Governance: Strict RESEARCH_ONLY; zero production changes.
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMRanker
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

backend_dir = Path(__file__).resolve().parents[3]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.research.data.store import ResearchDataStore
from app.research.walk_forward import (
    walk_forward_expanding,
    PURGE_DAYS,
    EMBARGO_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BR005")

DATA_DIR = backend_dir / "data"

FEATURE_COLS = [
    "dd_from_30d_high", "dd_from_90d_high", "dd_from_180d_high", "dd_from_365d_high",
    "up_from_30d_low", "up_from_90d_low", "up_from_180d_low", "up_from_365d_low",
    "dist_from_ath", "dist_from_atl",
    "dist_ema20", "dist_ema50", "dist_ema200", "ema20_slope_5d",
    "mom_7d", "mom_14d", "mom_30d", "mom_60d", "mom_90d",
    "rsi14", "rsi_change_7d", "macd_hist", "macd_accel_3d",
    "adx14", "adx_change_7d",
    "lower_low_count_20d", "higher_low_count_20d", "lower_high_count_20d", "higher_high_count_20d",
    "trend_persistence_20d",
    "rvol20", "vol_zscore_30d", "vol_price_divergence", "taker_buy_ratio", "amihud_illiquidity_proxy",
    "realized_vol_14d", "vol_change_14d", "atr_pct_14d", "bb_width", "bb_percentile_90d",
    "rel_strength_btc_7d", "rel_strength_btc_14d", "rel_strength_eth_7d", "rel_strength_eth_14d",
    "market_breadth_proxy"
]

HORIZONS = [14, 7, 30]  # 14D primary, 7D and 30D secondary

# ---------------------------------------------------------------------------
# Data Loading & Forward Return Calculation
# ---------------------------------------------------------------------------

def load_and_preprocess_dataset(store: ResearchDataStore) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    symbols = store.available_symbols("BR-002", "features")
    logger.info(f"Loading features and computing forward returns for {len(symbols)} symbols...")

    panels = []
    for sym in symbols:
        raw = store.load_raw(sym)
        feat = store.load_features("BR-002", sym)
        if raw is None or feat is None:
            continue

        feat["symbol"] = sym
        closes = raw["close"].values
        n_bars = len(closes)

        # Forward arithmetic returns: Close[t+h] / Close[t] - 1
        for h in HORIZONS:
            fwd_ret = np.full(n_bars, np.nan)
            if n_bars > h:
                fwd_ret[:-h] = (closes[h:] / closes[:-h]) - 1.0
            feat[f"fwd_ret_{h}d"] = fwd_ret

        panels.append(feat)

    full_df = pd.concat(panels, ignore_index=True)
    full_df["open_time_dt"] = pd.to_datetime(full_df["open_time"], unit="ms", utc=True)
    full_df = full_df.sort_values(["open_time_dt", "symbol"]).reset_index(drop=True)

    min_date = full_df["open_time_dt"].min().floor("D")
    max_date = full_df["open_time_dt"].max().floor("D")
    calendar_timeline = pd.date_range(min_date, max_date, freq="D", tz="UTC")

    logger.info(f"Loaded {len(full_df):,} observations from {min_date.date()} to {max_date.date()}.")
    return full_df, calendar_timeline

# ---------------------------------------------------------------------------
# Statistical Inference: HAC / Newey-West & Circular Block Bootstrap
# ---------------------------------------------------------------------------

def compute_hac_stats(series: np.ndarray, lag: int) -> Tuple[float, float, float]:
    """
    Computes sample mean, Newey-West HAC standard error, and HAC t-statistic.
    """
    T = len(series)
    if T <= lag or T == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(series))
    deviations = series - mean_val

    # Gamma_0 (sample variance)
    gamma_0 = float(np.mean(deviations ** 2))

    # Weighted autocovariances (Bartlett kernel)
    weighted_sum = 0.0
    for l in range(1, lag + 1):
        weight = 1.0 - (l / (lag + 1.0))
        gamma_l = float(np.mean(deviations[l:] * deviations[:-l]))
        weighted_sum += 2.0 * weight * gamma_l

    var_hac = (gamma_0 + weighted_sum) / T
    se_hac = math.sqrt(max(var_hac, 1e-12))
    t_hac = mean_val / se_hac
    return mean_val, se_hac, t_hac


def circular_block_bootstrap_ci(series: np.ndarray, block_len: int = 14, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """
    Computes 95% circular block bootstrap confidence interval for the mean.
    """
    N = len(series)
    if N < block_len * 2:
        return float(np.mean(series)), float(np.mean(series))

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(N / block_len))
    means = []

    for _ in range(n_boot):
        start_indices = rng.integers(0, N, size=n_blocks)
        sample = []
        for s in start_indices:
            idx = (np.arange(s, s + block_len)) % N
            sample.extend(series[idx])
        means.append(float(np.mean(sample[:N])))

    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))
    return ci_low, ci_high

# ---------------------------------------------------------------------------
# Phase 1: Cross-Sectional Ranking Evaluator
# ---------------------------------------------------------------------------

def evaluate_cross_sectional_ranking(
    test_df: pd.DataFrame,
    scores_dict: Dict[str, np.ndarray],
    target_col: str,
    horizon_days: int
) -> Dict[str, Any]:
    """
    Computes daily cross-sectional Spearman Rank IC, HAC t-stat, bootstrap CI,
    and full quintile return monotonicity across out-of-sample test days.
    """
    test_dates = test_df["open_time_dt"].dt.floor("D").values
    unique_dates = np.unique(test_dates)

    results_by_model = {}

    for model_name, scores in scores_dict.items():
        daily_ics = []
        daily_q_returns = {1: [], 2: [], 3: [], 4: [], 5: []}

        for d in unique_dates:
            mask = (test_dates == d)
            sub_scores = scores[mask]
            sub_returns = test_df.loc[mask, target_col].values

            valid_mask = ~np.isnan(sub_scores) & ~np.isnan(sub_returns)
            if np.sum(valid_mask) < 15:  # Need minimum cross-sectional breadth
                continue

            s_v = sub_scores[valid_mask]
            r_v = sub_returns[valid_mask]

            # Spearman Rank Correlation
            r_ic, _ = spearmanr(s_v, r_v)
            if not np.isnan(r_ic):
                daily_ics.append(float(r_ic))

            # Quintile forward returns
            try:
                # Rank into 5 discrete bins (1 to 5) using first-rank to prevent duplicate edge errors
                s_ser = pd.Series(s_v)
                ranks = pd.qcut(s_ser.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
                for q_val in range(1, 6):
                    q_mask = (ranks.values == q_val)
                    if np.sum(q_mask) > 0:
                        daily_q_returns[q_val].append(float(np.mean(r_v[q_mask])))
            except Exception:
                pass

        daily_ic_arr = np.array(daily_ics)
        lag_nw = horizon_days + 1  # L = h + 1
        mean_ic, se_hac, t_hac = compute_hac_stats(daily_ic_arr, lag=lag_nw)
        ci_low, ci_high = circular_block_bootstrap_ci(daily_ic_arr, block_len=horizon_days, n_boot=1000)

        std_ic = float(np.std(daily_ic_arr)) if len(daily_ic_arr) > 0 else 0.0
        ir_ic = (mean_ic / std_ic) * math.sqrt(365.0 / horizon_days) if std_ic > 1e-9 else 0.0

        # Mean Quintile Returns
        q_means = {q: float(np.mean(daily_q_returns[q])) if len(daily_q_returns[q]) > 0 else 0.0 for q in range(1, 6)}
        
        # Check strict full monotonicity: Q1 < Q2 < Q3 < Q4 < Q5
        is_strictly_monotonic = (
            q_means[1] < q_means[2] < q_means[3] < q_means[4] < q_means[5]
        )
        q5_q1_spread = q_means[5] - q_means[1]

        results_by_model[model_name] = {
            "n_days": len(daily_ic_arr),
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ir_ic": ir_ic,
            "se_hac": se_hac,
            "t_hac": t_hac,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "q_means": q_means,
            "is_strictly_monotonic": is_strictly_monotonic,
            "q5_q1_spread": q5_q1_spread
        }

    return results_by_model

# ---------------------------------------------------------------------------
# Full Walk-Forward Evaluation Pipeline for BR-005
# ---------------------------------------------------------------------------

def run_br005_phase1_study():
    store = ResearchDataStore(DATA_DIR)
    full_df, calendar_timeline = load_and_preprocess_dataset(store)

    folds = walk_forward_expanding(
        calendar_timeline,
        min_train_days=180,
        step_days=90,
        purge_days=PURGE_DAYS,
        embargo_days=EMBARGO_DAYS
    )

    study_results_by_horizon = {}

    for h in HORIZONS:
        target_col = f"fwd_ret_{h}d"
        is_primary = (h == 14)
        logger.info(f"\n=======================================================")
        logger.info(f"Evaluating Horizon: {h}D ({'PRIMARY' if is_primary else 'SECONDARY'}) Target: {target_col}")
        logger.info(f"=======================================================")

        valid_df = full_df.dropna(subset=[target_col]).copy()

        pooled_test_records = []
        pooled_scores = {
            "B0_Random": [],
            "B1_Momentum": [],
            "B2_Reversal": [],
            "Model1_Ridge": [],
            "Model2_LightGBM_Reg": [],
            "Model3_LambdaRank": []
        }

        fold_reports = []

        for fold in folds:
            train_mask = (valid_df["open_time_dt"] >= fold.train_start) & (valid_df["open_time_dt"] <= fold.train_end)
            test_mask  = (valid_df["open_time_dt"] >= fold.test_start)  & (valid_df["open_time_dt"] <= fold.test_end)

            train_data = valid_df[train_mask].sort_values("open_time_dt").copy()
            test_data  = valid_df[test_mask].sort_values("open_time_dt").copy()

            if len(train_data) < 500 or len(test_data) < 100:
                continue

            X_train = train_data[FEATURE_COLS].fillna(0.0).values
            y_train = train_data[target_col].values

            X_test  = test_data[FEATURE_COLS].fillna(0.0).values
            y_test  = test_data[target_col].values

            # ---------------------------------------------------------------
            # Baselines (Strictly Preregistered Directions)
            # ---------------------------------------------------------------
            # B0: Uniform random [0, 1]
            rng = np.random.default_rng(42 + fold.fold_index)
            s_b0 = rng.uniform(0.0, 1.0, size=len(test_data))

            # B1: Simple Momentum (higher momentum = higher predicted rank)
            mom_col = f"mom_{h}d" if f"mom_{h}d" in test_data.columns else "mom_14d"
            s_b1 = test_data[mom_col].fillna(0.0).values

            # B2: Simple Reversal / Dip-Buying (-dd_from_90d_high: deepest dip = highest rank)
            s_b2 = -test_data["dd_from_90d_high"].fillna(0.0).values

            # ---------------------------------------------------------------
            # Model 1: Cross-Sectional Ridge Regression (L2)
            # ---------------------------------------------------------------
            ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=100.0, random_state=42))
            ridge_pipe.fit(X_train, y_train)
            s_ridge = ridge_pipe.predict(X_test)

            # ---------------------------------------------------------------
            # Model 2: LightGBM Forward Return Regressor
            # ---------------------------------------------------------------
            lgb_reg = LGBMRegressor(
                n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1
            )
            lgb_reg.fit(X_train, y_train)
            s_lgb_reg = lgb_reg.predict(X_test)

            # ---------------------------------------------------------------
            # Model 3: LightGBM Pairwise LambdaRank
            # Trained strictly on training fold forward returns.
            # Daily queries grouped by calendar date.
            # ---------------------------------------------------------------
            train_dates = train_data["open_time_dt"].dt.floor("D")
            train_groups = train_data.groupby(train_dates, sort=False).size().values

            # Relevance labels: convert forward returns into 5 integer rank tiers [0..4] per day
            # using first-rank to prevent duplicate edge errors
            train_data["rel_rank"] = train_data.groupby(train_dates, sort=False)[target_col].transform(
                lambda s: pd.qcut(s.rank(method="first"), 5, labels=[0, 1, 2, 3, 4]).astype(int)
            )
            y_train_rank = train_data["rel_rank"].values

            lgb_ranker = LGBMRanker(
                objective="lambdarank",
                n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1
            )
            lgb_ranker.fit(X_train, y_train_rank, group=train_groups)
            s_lgb_ranker = lgb_ranker.predict(X_test)

            fold_scores = {
                "B0_Random": s_b0,
                "B1_Momentum": s_b1,
                "B2_Reversal": s_b2,
                "Model1_Ridge": s_ridge,
                "Model2_LightGBM_Reg": s_lgb_reg,
                "Model3_LambdaRank": s_lgb_ranker
            }

            # Evaluate fold metrics
            fold_eval = evaluate_cross_sectional_ranking(test_data, fold_scores, target_col, h)
            fold_reports.append({
                "fold_idx": fold.fold_index,
                "train_dates": f"{fold.train_start.date()} -> {fold.train_end.date()}",
                "test_dates": f"{fold.test_start.date()} -> {fold.test_end.date()}",
                "n_train": len(train_data),
                "n_test": len(test_data),
                "models": fold_eval
            })

            # Accumulate pooled test records
            pooled_test_records.append(test_data[["open_time_dt", "symbol", target_col]])
            for k in pooled_scores:
                pooled_scores[k].extend(fold_scores[k])

        pooled_test_df = pd.concat(pooled_test_records, ignore_index=True)
        for k in pooled_scores:
            pooled_scores[k] = np.array(pooled_scores[k])

        # Evaluate pooled performance
        pooled_eval = evaluate_cross_sectional_ranking(pooled_test_df, pooled_scores, target_col, h)

        # Check Phase 1 Predictive Gate for Primary Horizon
        m3_pooled = pooled_eval["Model3_LambdaRank"]
        folds_positive_ic = sum(
            1 for fld in fold_reports if fld["models"]["Model3_LambdaRank"]["mean_ic"] > 0.0
        )
        gate1_strength = (m3_pooled["mean_ic"] >= 0.030) and (m3_pooled["t_hac"] >= 2.00) and (m3_pooled["bootstrap_ci_low"] > 0.0)
        gate2_monotonic = m3_pooled["is_strictly_monotonic"] and (m3_pooled["q5_q1_spread"] > 0.0)
        gate3_temporal = (folds_positive_ic >= 3)
        phase1_passed = gate1_strength and gate2_monotonic and gate3_temporal

        study_results_by_horizon[f"{h}d"] = {
            "horizon_days": h,
            "is_primary": is_primary,
            "target_col": target_col,
            "total_test_samples": len(pooled_test_df),
            "fold_reports": fold_reports,
            "pooled_eval": pooled_eval,
            "phase1_gate": {
                "gate1_strength": bool(gate1_strength),
                "gate2_monotonic": bool(gate2_monotonic),
                "gate3_temporal": bool(gate3_temporal),
                "phase1_passed": bool(phase1_passed),
                "m3_mean_ic": m3_pooled["mean_ic"],
                "m3_t_hac": m3_pooled["t_hac"],
                "m3_boot_ci": [m3_pooled["bootstrap_ci_low"], m3_pooled["bootstrap_ci_high"]],
                "m3_q_means": m3_pooled["q_means"],
                "m3_spread": m3_pooled["q5_q1_spread"],
                "folds_positive_ic": folds_positive_ic
            },
            # Store data for Phase 2 simulation
            "_test_df": pooled_test_df,
            "_scores": pooled_scores
        }

    # -----------------------------------------------------------------------
    # Phase 2 Gate Check: Only execute if Primary 14D passed Phase 1
    # -----------------------------------------------------------------------
    primary_14d_passed = study_results_by_horizon["14d"]["phase1_gate"]["phase1_passed"]
    logger.info(f"\n=======================================================")
    logger.info(f"Phase 1 Primary 14D Gate Passed: {primary_14d_passed}")
    logger.info(f"=======================================================")

    phase2_results = {}
    if primary_14d_passed:
        logger.info("Executing Phase 2 Friction-Adjusted Portfolio Simulation...")
        for h in HORIZONS:
            h_key = f"{h}d"
            h_data = study_results_by_horizon[h_key]
            p2_res = run_phase2_portfolio_simulation(
                h_data["_test_df"], h_data["_scores"], h_data["target_col"], h, fee_rate=0.0040
            )
            phase2_results[h_key] = p2_res
    else:
        logger.warning("Phase 1 Primary 14D Gate FAILED. Phase 2 Portfolio Simulation will NOT be executed.")

    # Clean internal dataframes from results dictionary before saving
    final_output = {
        "study_results_by_horizon": {},
        "phase2_portfolio_simulation": phase2_results
    }
    for h_key, val in study_results_by_horizon.items():
        clean_val = {k: v for k, v in val.items() if not k.startswith("_")}
        final_output["study_results_by_horizon"][h_key] = clean_val

    return final_output

# ---------------------------------------------------------------------------
# Phase 2: Friction-Adjusted Portfolio Simulation Engine
# ---------------------------------------------------------------------------

def simulate_strategy_rebalancing(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    target_col: str,
    h_days: int,
    strategy_type: str = "Q5",  # 'Q5', 'D10', 'EW'
    fee_rate: float = 0.0040
) -> Dict[str, Any]:
    """
    Simulates non-overlapping periodic rebalanced portfolio with exact 0.40% round-trip friction.
    """
    test_dates = test_df["open_time_dt"].dt.floor("D")
    unique_dates = np.sort(test_dates.unique())

    # Step by h_days for non-overlapping holding periods
    rebalance_dates = unique_dates[::h_days]

    period_records = []
    prev_weights: Dict[str, float] = {}
    wealth_curve = [1.0]

    for reb_date in rebalance_dates:
        mask = (test_dates == reb_date)
        sub_df = test_df.loc[mask]
        sub_scores = scores[mask]
        sub_ret = sub_df[target_col].values
        symbols = sub_df["symbol"].values

        valid_m = ~np.isnan(sub_scores) & ~np.isnan(sub_ret)
        if np.sum(valid_m) < 15:
            continue

        sym_v = symbols[valid_m]
        sc_v = sub_scores[valid_m]
        ret_v = sub_ret[valid_m]

        N = len(sym_v)
        # Determine target portfolio selection
        if strategy_type == "EW":
            selected_idx = np.arange(N)
        elif strategy_type == "D10":
            k_top = max(1, int(math.ceil(N * 0.10)))
            selected_idx = np.argsort(sc_v)[-k_top:]
        elif strategy_type == "Q5":
            k_top = max(1, int(math.ceil(N * 0.20)))
            selected_idx = np.argsort(sc_v)[-k_top:]
        elif strategy_type == "Q1":
            k_bot = max(1, int(math.ceil(N * 0.20)))
            selected_idx = np.argsort(sc_v)[:k_bot]
        else:
            selected_idx = np.arange(N)

        target_syms = sym_v[selected_idx]
        target_ret = ret_v[selected_idx]
        k_count = len(target_syms)

        # Equal weighting across selected basket
        new_weights = {sym: (1.0 / k_count) for sym in target_syms}

        # Calculate portfolio turnover relative to previous rebalance
        all_syms = set(prev_weights.keys()).union(set(new_weights.keys()))
        turnover = 0.5 * sum(abs(new_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms)
        prev_weights = new_weights

        # Friction cost
        friction = turnover * fee_rate

        # Gross basket return
        gross_basket_ret = float(np.mean(target_ret))
        # Net period return after friction
        net_ret = float((1.0 + gross_basket_ret) * (1.0 - friction) - 1.0)

        # Update wealth
        new_wealth = wealth_curve[-1] * (1.0 + net_ret)
        wealth_curve.append(new_wealth)

        period_records.append({
            "rebalance_date": str(pd.Timestamp(reb_date).date()),
            "n_assets": k_count,
            "turnover": turnover,
            "friction": friction,
            "gross_return": gross_basket_ret,
            "net_return": net_ret,
            "wealth": new_wealth
        })

    if not period_records:
        return {}

    net_returns = np.array([r["net_return"] for r in period_records])
    turnovers = np.array([r["turnover"] for r in period_records])
    wealth_arr = np.array(wealth_curve)

    # Compute drawdowns
    running_max = np.maximum.accumulate(wealth_arr)
    drawdowns = (running_max - wealth_arr) / running_max
    max_drawdown = float(np.max(drawdowns))

    # Annualization factor
    n_periods = len(net_returns)
    periods_per_year = 365.0 / h_days
    total_days = n_periods * h_days

    cum_return = float(wealth_arr[-1] - 1.0)
    cagr_net = float((wealth_arr[-1] ** (365.0 / total_days)) - 1.0) if total_days > 0 and wealth_arr[-1] > 0 else 0.0

    mean_per_ret = float(np.mean(net_returns))
    std_per_ret = float(np.std(net_returns))
    ann_vol = float(std_per_ret * math.sqrt(periods_per_year))

    sharpe_net = float((mean_per_ret / std_per_ret) * math.sqrt(periods_per_year)) if std_per_ret > 1e-9 else 0.0

    downside_rets = np.minimum(net_returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside_rets ** 2)))
    sortino_net = float((mean_per_ret / downside_dev) * math.sqrt(periods_per_year)) if downside_dev > 1e-9 else 0.0

    calmar_net = float(cagr_net / max_drawdown) if max_drawdown > 1e-9 else 0.0
    avg_turnover = float(np.mean(turnovers))

    return {
        "strategy": strategy_type,
        "n_periods": n_periods,
        "total_days": total_days,
        "cum_return": cum_return,
        "cagr_net": cagr_net,
        "ann_vol": ann_vol,
        "sharpe_net": sharpe_net,
        "sortino_net": sortino_net,
        "max_drawdown": max_drawdown,
        "calmar_net": calmar_net,
        "avg_turnover": avg_turnover,
        "period_records": period_records
    }

def run_phase2_portfolio_simulation(
    test_df: pd.DataFrame,
    scores_dict: Dict[str, np.ndarray],
    target_col: str,
    h_days: int,
    fee_rate: float = 0.0040
) -> Dict[str, Any]:
    """
    Executes Phase 2 simulations across benchmark and candidate portfolios.
    """
    sim_configs = [
        ("Benchmark_EW", scores_dict["B0_Random"], "EW"),
        ("B1_Momentum_Q5", scores_dict["B1_Momentum"], "Q5"),
        ("B2_Reversal_Q5", scores_dict["B2_Reversal"], "Q5"),
        ("Model1_Ridge_Q5", scores_dict["Model1_Ridge"], "Q5"),
        ("Model2_LightGBM_Reg_Q5", scores_dict["Model2_LightGBM_Reg"], "Q5"),
        ("Model3_LambdaRank_Q5", scores_dict["Model3_LambdaRank"], "Q5"),
        ("Model3_LambdaRank_D10", scores_dict["Model3_LambdaRank"], "D10")
    ]

    p2_results = {}
    for name, sc, stype in sim_configs:
        res = simulate_strategy_rebalancing(
            test_df, sc, target_col, h_days, strategy_type=stype, fee_rate=fee_rate
        )
        p2_results[name] = res

    return p2_results

if __name__ == "__main__":
    results = run_br005_phase1_study()
    out_json = backend_dir / "br005_full_results_raw.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"BR-005 Full study complete. Results saved to {out_json}")
