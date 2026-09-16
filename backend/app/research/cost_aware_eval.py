import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class CostAwareEvaluator:
    """
    Cost-Aware Decision & Return Evaluator for BR-001 (Section 8.1 & 8.2).
    Evaluates net trading performance after:
    R_net = R_gross - fee_roundtrip - spread_roundtrip - slippage_roundtrip - impact
    Enforces decision gating: If EV_net <= 0 -> NO_TRADE.
    """

    DEFAULT_FRICTION = {
        "taker_fee_one_way": 0.0010,       # 0.10%
        "spread_one_way": 0.0005,          # 0.05%
        "slippage_one_way": 0.0005,        # 0.05%
        "impact_one_way": 0.0000,          # 0.00% for standard sizing
    }

    @classmethod
    def calculate_roundtrip_friction(cls, custom_friction: Optional[Dict[str, float]] = None) -> float:
        cfg = cls.DEFAULT_FRICTION.copy()
        if custom_friction:
            cfg.update(custom_friction)
        one_way = (
            cfg.get("taker_fee_one_way", 0.0010) +
            cfg.get("spread_one_way", 0.0005) +
            cfg.get("slippage_one_way", 0.0005) +
            cfg.get("impact_one_way", 0.0)
        )
        return float(2.0 * one_way)

    @classmethod
    def evaluate_trades(
        cls,
        signals: np.ndarray,
        forward_returns: np.ndarray,
        prob_threshold: float = 0.55,
        probabilities: Optional[np.ndarray] = None,
        custom_friction: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a signal or probability series against forward returns.
        Signals: boolean or 1/0 indicating trade entry at time t.
        """
        valid = ~np.isnan(forward_returns)
        if probabilities is not None:
            valid = valid & ~np.isnan(probabilities)
            trade_mask = valid & (probabilities >= prob_threshold)
        else:
            valid = valid & ~np.isnan(signals)
            trade_mask = valid & (signals > 0)

        trade_count = int(np.sum(trade_mask))
        friction = cls.calculate_roundtrip_friction(custom_friction)

        if trade_count < 3:
            return {
                "trade_count": trade_count,
                "gross_return": 0.0,
                "net_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_expectancy": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "decision_gate": "NO_TRADE",
                "gate_reason": "Insufficient trade occurrences to establish statistical validity"
            }

        gross_rets = forward_returns[trade_mask]
        net_rets = gross_rets - friction

        wins = net_rets[net_rets > 0]
        losses = net_rets[net_rets <= 0]

        win_rate = len(wins) / len(net_rets) if len(net_rets) > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0

        net_expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        total_net_ret = float(np.sum(net_rets))
        total_gross_ret = float(np.sum(gross_rets))

        gross_gain = float(np.sum(gross_rets[gross_rets > 0])) if np.sum(gross_rets > 0) > 0 else 1e-9
        gross_loss = float(abs(np.sum(gross_rets[gross_rets < 0]))) if np.sum(gross_rets < 0) > 0 else 1e-9
        profit_factor = gross_gain / gross_loss

        # Annualized Sharpe (assuming 14D holding periods ~ 26 trades/yr)
        ret_std = float(np.std(net_rets))
        sharpe = (float(np.mean(net_rets)) / (ret_std + 1e-9)) * np.sqrt(26) if ret_std > 0 else 0.0

        # Downside deviation for Sortino
        neg_rets = net_rets[net_rets < 0]
        downside_std = float(np.std(neg_rets)) if len(neg_rets) > 1 else ret_std
        sortino = (float(np.mean(net_rets)) / (downside_std + 1e-9)) * np.sqrt(26) if downside_std > 0 else 0.0

        # Cumulative drawdown
        cum_equity = np.cumprod(1.0 + net_rets)
        rolling_max = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - rolling_max) / (rolling_max + 1e-9)
        max_dd = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        # Decision gate logic
        if net_expectancy <= 0.0:
            decision_gate = "NO_TRADE"
            gate_reason = f"EV_net ({net_expectancy:.4f}) <= 0 after {friction*100:.2f}% roundtrip friction"
        elif sharpe < 0.5:
            decision_gate = "NO_TRADE"
            gate_reason = f"Net Sharpe ({sharpe:.2f}) < minimum hurdle of 0.50"
        elif max_dd > 0.25:
            decision_gate = "REDUCE_SIZE"
            gate_reason = f"Max drawdown ({max_dd*100:.1f}%) exceeds 25% threshold"
        else:
            decision_gate = "ELIGIBLE"
            gate_reason = f"EV_net positive (+{net_expectancy*100:.2f}%), net Sharpe {sharpe:.2f}"

        return {
            "trade_count": trade_count,
            "gross_return": round(total_gross_ret, 4),
            "net_return": round(total_net_ret, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2),
            "net_expectancy": round(net_expectancy, 4),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown": round(max_dd, 4),
            "roundtrip_friction": round(friction, 4),
            "decision_gate": decision_gate,
            "gate_reason": gate_reason
        }

    @classmethod
    def run_friction_sensitivity(
        cls,
        forward_returns: np.ndarray,
        trade_mask: np.ndarray,
        frictions: Optional[List[float]] = None
    ) -> List[Dict[str, float]]:
        """
        Computes Net Return and Net Expectancy across varying friction hurdles:
        e.g. 0.1%, 0.2%, 0.4%, 0.6%, 0.8%, 1.0%, 1.5%.
        """
        if frictions is None:
            frictions = [0.0010, 0.0020, 0.0040, 0.0060, 0.0080, 0.0100, 0.0150]

        gross_rets = forward_returns[trade_mask & ~np.isnan(forward_returns)]
        if len(gross_rets) == 0:
            return []

        results = []
        for f in frictions:
            net_rets = gross_rets - f
            net_exp = float(np.mean(net_rets))
            wins = net_rets[net_rets > 0]
            wr = len(wins) / len(net_rets) if len(net_rets) > 0 else 0.0
            results.append({
                "friction_pct": round(f * 100, 2),
                "net_expectancy": round(net_exp, 4),
                "win_rate": round(wr, 4),
                "net_sum": round(float(np.sum(net_rets)), 4)
            })

        return results
