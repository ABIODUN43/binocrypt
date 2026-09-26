"""
Robustness Audit Engine for BR-005 (14D Primary and 30D Secondary)
Generates exhaustive empirical metrics for BR-005-ROBUSTNESS-AUDIT.md
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.research.data.store import ResearchDataStore
from app.research.walk_forward import walk_forward_expanding
from app.research.br005.study_br005 import (
    FEATURE_COLS,
    HORIZONS,
    load_and_preprocess_dataset,
    simulate_strategy_rebalancing,
)

def run_audit():
    store = ResearchDataStore(backend_dir / "data")
    panel_df, date_index = load_and_preprocess_dataset(store)

    # We need to re-run or inspect predictions across folds for 14D and 30D
    # Let's run walk-forward predictions for Model 3 LambdaRank to retain asset-level predictions
    folds = walk_forward_expanding(
        date_index,
        min_train_days=180,
        step_days=90,
        purge_days=90,
        embargo_days=45,
    )
    print(f"Generated {len(folds)} walk-forward folds.")

    audit_results = {}

    for h in [14, 30]:
        h_key = f"{h}d"
        target_col = f"fwd_ret_{h}d"
        print(f"\n==========================================")
        print(f"AUDITING HORIZON {h_key} (Target: {target_col})")
        print(f"==========================================")

        # Collect fold predictions
        fold_test_dfs = []
        for f_idx, fold in enumerate(folds):
            print(f"Training Model 3 LambdaRank on Fold {f_idx} for {h_key}...")
            # Filter clean data
            train_mask = (panel_df["open_time_dt"] >= fold.train_start) & (panel_df["open_time_dt"] <= fold.train_end)
            test_mask = (panel_df["open_time_dt"] >= fold.test_start) & (panel_df["open_time_dt"] <= fold.test_end)

            train_df = panel_df.loc[train_mask].dropna(subset=FEATURE_COLS + [target_col]).copy()
            test_df = panel_df.loc[test_mask].dropna(subset=FEATURE_COLS + [target_col]).copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            # LambdaRank requires integer relevance grades per date
            train_dates = train_df["open_time_dt"].dt.floor("D")
            train_df["_date"] = train_dates
            groups = []
            rel_grades = []
            for dt, grp in train_df.groupby("_date", sort=True):
                g_size = len(grp)
                groups.append(g_size)
                # 5 relevance grades [0, 1, 2, 3, 4] based on quintile of forward return
                if g_size >= 5:
                    q = pd.qcut(grp[target_col].rank(method="first"), 5, labels=[0, 1, 2, 3, 4]).astype(int)
                else:
                    q = pd.Series(0, index=grp.index)
                rel_grades.extend(q.values)

            from lightgbm import LGBMRanker
            model = LGBMRanker(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                random_state=42 + f_idx,
                n_jobs=-1,
                verbosity=-1
            )
            model.fit(train_df[FEATURE_COLS], rel_grades, group=groups)
            test_scores = model.predict(test_df[FEATURE_COLS])
            test_df["score"] = test_scores
            test_df["fold"] = f_idx
            fold_test_dfs.append(test_df)

        full_test_df = pd.concat(fold_test_dfs, axis=0).reset_index(drop=True)
        print(f"Total test rows for {h_key}: {len(full_test_df)}")

        # 1. Fold-by-Fold Economic Performance
        fold_metrics = {}
        for f_idx in range(len(folds)):
            f_df = full_test_df[full_test_df["fold"] == f_idx].copy()
            if len(f_df) == 0:
                continue
            sim_q5 = simulate_strategy_rebalancing(f_df, f_df["score"].values, target_col, h, "Q5", 0.0040)
            sim_d10 = simulate_strategy_rebalancing(f_df, f_df["score"].values, target_col, h, "D10", 0.0040)
            sim_ew = simulate_strategy_rebalancing(f_df, f_df["score"].values, target_col, h, "EW", 0.0040)
            fold_metrics[f"fold_{f_idx}"] = {
                "n_periods": sim_q5.get("n_periods", 0),
                "q5_return": sim_q5.get("cum_return", 0.0),
                "q5_sharpe": sim_q5.get("sharpe_net", 0.0),
                "q5_maxdd": sim_q5.get("max_drawdown", 0.0),
                "q5_turnover": sim_q5.get("avg_turnover", 0.0),
                "d10_return": sim_d10.get("cum_return", 0.0),
                "d10_sharpe": sim_d10.get("sharpe_net", 0.0),
                "ew_return": sim_ew.get("cum_return", 0.0),
                "ew_sharpe": sim_ew.get("sharpe_net", 0.0),
                "ew_maxdd": sim_ew.get("max_drawdown", 0.0),
            }

        # 2. Detailed Period-by-Period History & Concentration
        # Simulate over all non-overlapping dates
        test_dates = full_test_df["open_time_dt"].dt.floor("D")
        unique_dates = np.sort(test_dates.unique())
        rebalance_dates = unique_dates[::h]

        period_records = []
        prev_q5_weights = {}
        prev_d10_weights = {}
        prev_ew_weights = {}

        q5_wealth = 1.0
        d10_wealth = 1.0
        ew_wealth = 1.0

        for reb_date in rebalance_dates:
            mask = (test_dates == reb_date)
            sub = full_test_df.loc[mask]
            if len(sub) < 15:
                continue

            syms = sub["symbol"].values
            scs = sub["score"].values
            rets = sub[target_col].values
            N = len(syms)

            # Q5
            k_q5 = max(1, int(math.ceil(N * 0.20)))
            idx_q5 = np.argsort(scs)[-k_q5:]
            syms_q5 = syms[idx_q5]
            rets_q5 = rets[idx_q5]
            w_q5 = {s: 1.0 / k_q5 for s in syms_q5}
            t_q5 = 0.5 * sum(abs(w_q5.get(s, 0.0) - prev_q5_weights.get(s, 0.0)) for s in set(w_q5).union(prev_q5_weights))
            f_q5 = t_q5 * 0.0040
            g_ret_q5 = float(np.mean(rets_q5))
            net_ret_q5 = float((1.0 + g_ret_q5) * (1.0 - f_q5) - 1.0)
            q5_wealth *= (1.0 + net_ret_q5)
            prev_q5_weights = w_q5

            # D10
            k_d10 = max(1, int(math.ceil(N * 0.10)))
            idx_d10 = np.argsort(scs)[-k_d10:]
            syms_d10 = syms[idx_d10]
            rets_d10 = rets[idx_d10]
            w_d10 = {s: 1.0 / k_d10 for s in syms_d10}
            t_d10 = 0.5 * sum(abs(w_d10.get(s, 0.0) - prev_d10_weights.get(s, 0.0)) for s in set(w_d10).union(prev_d10_weights))
            f_d10 = t_d10 * 0.0040
            g_ret_d10 = float(np.mean(rets_d10))
            net_ret_d10 = float((1.0 + g_ret_d10) * (1.0 - f_d10) - 1.0)
            d10_wealth *= (1.0 + net_ret_d10)
            prev_d10_weights = w_d10

            # EW
            w_ew = {s: 1.0 / N for s in syms}
            t_ew = 0.5 * sum(abs(w_ew.get(s, 0.0) - prev_ew_weights.get(s, 0.0)) for s in set(w_ew).union(prev_ew_weights))
            f_ew = t_ew * 0.0040
            g_ret_ew = float(np.mean(rets))
            net_ret_ew = float((1.0 + g_ret_ew) * (1.0 - f_ew) - 1.0)
            ew_wealth *= (1.0 + net_ret_ew)
            prev_ew_weights = w_ew

            # Store asset level details for this rebalance period
            asset_details = [{"symbol": s, "score": float(sc), "return": float(r)} for s, sc, r in zip(syms_q5, scs[idx_q5], rets_q5)]

            period_records.append({
                "rebalance_date": str(pd.Timestamp(reb_date).date()),
                "q5_gross_return": g_ret_q5,
                "q5_net_return": net_ret_q5,
                "q5_turnover": t_q5,
                "q5_friction": f_q5,
                "q5_wealth": q5_wealth,
                "d10_net_return": net_ret_d10,
                "d10_wealth": d10_wealth,
                "ew_net_return": net_ret_ew,
                "ew_wealth": ew_wealth,
                "excess_q5_vs_ew": net_ret_q5 - net_ret_ew,
                "q5_assets": asset_details,
            })

        # 3. Batting Average / Period-by-Period Excess Return
        excess_returns = [p["excess_q5_vs_ew"] for p in period_records]
        win_count = sum(1 for e in excess_returns if e > 0)
        total_p = len(excess_returns)
        batting_avg = win_count / total_p if total_p > 0 else 0.0

        # Market condition breakdown
        up_market_periods = [p for p in period_records if p["ew_net_return"] > 0]
        down_market_periods = [p for p in period_records if p["ew_net_return"] <= 0]

        up_market_wins = sum(1 for p in up_market_periods if p["excess_q5_vs_ew"] > 0)
        down_market_wins = sum(1 for p in down_market_periods if p["excess_q5_vs_ew"] > 0)

        up_market_win_rate = up_market_wins / len(up_market_periods) if up_market_periods else 0.0
        down_market_win_rate = down_market_wins / len(down_market_periods) if down_market_periods else 0.0

        # 4. Friction Sensitivity Sweep
        fee_rates = [0.0000, 0.0010, 0.0020, 0.0040, 0.0060, 0.0080, 0.0100, 0.0150]
        friction_sweep = {}
        for fee in fee_rates:
            sim_fee = simulate_strategy_rebalancing(
                full_test_df, full_test_df["score"].values, target_col, h, "Q5", fee
            )
            friction_sweep[f"fee_{int(fee*10000)}bps"] = {
                "cum_return": sim_fee.get("cum_return", 0.0),
                "sharpe_net": sim_fee.get("sharpe_net", 0.0),
                "max_drawdown": sim_fee.get("max_drawdown", 0.0),
            }

        # 5. Outlier Sensitivity (Removing top 1, 2, 3 periods and worst periods)
        period_net_rets = np.array([p["q5_net_return"] for p in period_records])
        sorted_indices = np.argsort(period_net_rets)

        def recompute_stats(rets_subset, h_days):
            wealth = np.cumprod(1.0 + rets_subset)
            total_ret = wealth[-1] - 1.0
            ppy = 365.0 / h_days
            mean_r = np.mean(rets_subset)
            std_r = np.std(rets_subset)
            sharpe = (mean_r / std_r) * math.sqrt(ppy) if std_r > 1e-9 else 0.0
            running_max = np.maximum.accumulate(np.insert(wealth, 0, 1.0))
            wealth_w0 = np.insert(wealth, 0, 1.0)
            dds = (running_max - wealth_w0) / running_max
            max_dd = float(np.max(dds))
            return {"cum_return": float(total_ret), "sharpe": float(sharpe), "max_dd": max_dd}

        # Remove top 1 period
        rets_no_top1 = np.delete(period_net_rets, sorted_indices[-1])
        # Remove top 2 periods
        rets_no_top2 = np.delete(period_net_rets, sorted_indices[-2:])
        # Remove top 3 periods
        rets_no_top3 = np.delete(period_net_rets, sorted_indices[-3:])
        # Remove worst 1 period
        rets_no_worst1 = np.delete(period_net_rets, sorted_indices[0])

        outlier_sensitivity = {
            "all_periods": recompute_stats(period_net_rets, h),
            "no_top1": recompute_stats(rets_no_top1, h),
            "no_top2": recompute_stats(rets_no_top2, h),
            "no_top3": recompute_stats(rets_no_top3, h),
            "no_worst1": recompute_stats(rets_no_worst1, h),
            "top1_period_detail": {
                "date": period_records[sorted_indices[-1]]["rebalance_date"],
                "q5_return": float(period_records[sorted_indices[-1]]["q5_net_return"]),
                "ew_return": float(period_records[sorted_indices[-1]]["ew_net_return"]),
            },
            "top2_period_detail": {
                "date": period_records[sorted_indices[-2]]["rebalance_date"],
                "q5_return": float(period_records[sorted_indices[-2]]["q5_net_return"]),
                "ew_return": float(period_records[sorted_indices[-2]]["ew_net_return"]),
            } if len(sorted_indices) > 1 else {},
            "worst1_period_detail": {
                "date": period_records[sorted_indices[0]]["rebalance_date"],
                "q5_return": float(period_records[sorted_indices[0]]["q5_net_return"]),
                "ew_return": float(period_records[sorted_indices[0]]["ew_net_return"]),
            }
        }

        # 6. Asset-Level Contribution Concentration
        # Tally each asset's contribution across all holding periods
        asset_pnl_counts = {}
        asset_total_return = {}
        for p in period_records:
            n_in_basket = len(p["q5_assets"])
            w = 1.0 / n_in_basket
            for item in p["q5_assets"]:
                sym = item["symbol"]
                r = item["return"]
                asset_pnl_counts[sym] = asset_pnl_counts.get(sym, 0) + 1
                asset_total_return[sym] = asset_total_return.get(sym, 0.0) + (w * r)

        top_contributing_assets = sorted(asset_total_return.items(), key=lambda x: x[1], reverse=True)[:10]
        worst_contributing_assets = sorted(asset_total_return.items(), key=lambda x: x[1])[:10]

        total_gross_alpha_sum = sum(asset_total_return.values())
        top3_contrib_sum = sum(v for k, v in top_contributing_assets[:3])
        top3_contrib_share = top3_contrib_sum / total_gross_alpha_sum if total_gross_alpha_sum > 0 else 0.0

        # Save audit bundle for this horizon
        audit_results[h_key] = {
            "horizon": h,
            "total_test_rows": len(full_test_df),
            "n_rebalance_periods": len(period_records),
            "fold_metrics": fold_metrics,
            "batting_average": {
                "total_periods": total_p,
                "win_count": win_count,
                "overall_win_rate": batting_avg,
                "up_market_periods": len(up_market_periods),
                "up_market_win_rate": up_market_win_rate,
                "down_market_periods": len(down_market_periods),
                "down_market_win_rate": down_market_win_rate,
            },
            "friction_sweep": friction_sweep,
            "outlier_sensitivity": outlier_sensitivity,
            "asset_concentration": {
                "unique_assets_selected": len(asset_pnl_counts),
                "top3_contrib_share": top3_contrib_share,
                "top_contributing_assets": top_contributing_assets,
                "worst_contributing_assets": worst_contributing_assets,
            },
            "period_records": [{k: v for k, v in pr.items() if k != "q5_assets"} for pr in period_records]
        }

    # Save to disk
    out_path = backend_dir / "br005_robustness_audit_raw.json"
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit complete. Saved results to {out_path}")

if __name__ == "__main__":
    run_audit()
