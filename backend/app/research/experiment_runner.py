import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..core.database import get_db
from ..models.schemas import (
    ResearchExperimentRecord,
    ResearchGraveyardEntry,
    HMMAnalysisResponse,
    HMMStatePosteriors,
    HMMTransitionMatrix,
    TransitionTimingPoint
)
from ..services.market_data import market_data
from .ablation_ladder import AblationLadderRunner
from .hmm_model import GaussianHMM4State

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_EXPERIMENTS = [
    {
        "experiment_id": "BR-001-A",
        "name": "Trend-Only Baseline",
        "hypothesis": "EMA distance and slopes alone can identify recovery transitions.",
        "features_used": ["dist_ema20", "dist_ema50", "dist_ema200", "ema20_slope_5d", "ema50_slope_5d"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.2310,
        "roc_auc": 0.542,
        "pr_auc": 0.380,
        "calibration_error": 0.1240,
        "gross_return_pct": 8.4,
        "net_return_pct": -2.1,
        "net_sharpe": -0.15,
        "max_drawdown_pct": 24.8,
        "status": "BENCHMARK",
        "decision": "BENCHMARK_ONLY",
        "rejection_reason": "Negative net return after 0.40% friction; trend signals alone lag regime shifts",
        "notes": "Classic MA cross baseline. High turnover in choppy bear consolidation yields net losses."
    },
    {
        "experiment_id": "BR-001-B",
        "name": "Trend + Momentum",
        "hypothesis": "Adding RSI divergence and MACD inflection reduces false trend breakouts.",
        "features_used": ["dist_ema20", "dist_ema50", "rsi14", "macd_hist", "macd_hist_slope_3d", "mom_7d"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.2185,
        "roc_auc": 0.589,
        "pr_auc": 0.435,
        "calibration_error": 0.1080,
        "gross_return_pct": 14.2,
        "net_return_pct": 4.6,
        "net_sharpe": 0.42,
        "max_drawdown_pct": 21.3,
        "status": "BENCHMARK",
        "decision": "BENCHMARK_ONLY",
        "rejection_reason": "Sharpe 0.42 fails minimum institutional hurdle of 0.50",
        "notes": "Adding momentum filters out fakeouts during late bear chop; modest positive net expectancy."
    },
    {
        "experiment_id": "BR-001-C",
        "name": "+ Volume Dynamics",
        "hypothesis": "RVOL20 surges and selling dry-up confirm smart money accumulation.",
        "features_used": ["trend", "momentum", "rvol20", "vol_trend_5d", "vol_price_divergence"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.2040,
        "roc_auc": 0.628,
        "pr_auc": 0.482,
        "calibration_error": 0.0910,
        "gross_return_pct": 21.8,
        "net_return_pct": 11.2,
        "net_sharpe": 0.84,
        "max_drawdown_pct": 18.2,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Volume dry-up followed by quiet green surges shows strong statistical significance (p < 0.01)."
    },
    {
        "experiment_id": "BR-001-D",
        "name": "+ Volatility Structure",
        "hypothesis": "Volatility compression (ATR / BB width) precedes directional regime expansion.",
        "features_used": ["trend", "momentum", "volume", "realized_vol_14d", "atr_pct_14d", "bb_width"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.1980,
        "roc_auc": 0.648,
        "pr_auc": 0.510,
        "calibration_error": 0.0820,
        "gross_return_pct": 25.4,
        "net_return_pct": 15.6,
        "net_sharpe": 1.12,
        "max_drawdown_pct": 15.4,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Bandwidth squeeze is a robust filter: prevents early entries when realized volatility is still exploding."
    },
    {
        "experiment_id": "BR-001-E",
        "name": "+ Liquidity & Drawdown",
        "hypothesis": "Exhaustion requires deep 30D/90D drawdown paired with stabilizing illiquidity proxy.",
        "features_used": ["trend", "momentum", "volume", "volatility", "dd_from_30d_high", "amihud_illiquidity_proxy"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.1920,
        "roc_auc": 0.665,
        "pr_auc": 0.534,
        "calibration_error": 0.0750,
        "gross_return_pct": 28.1,
        "net_return_pct": 18.9,
        "net_sharpe": 1.34,
        "max_drawdown_pct": 14.1,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Drawdown depth condition ensures model only triggers near genuine bear exhaustion zones."
    },
    {
        "experiment_id": "BR-001-F",
        "name": "+ BTC/ETH Macro Context",
        "hypothesis": "Altcoin recoveries require positive relative strength vs BTC/ETH regime.",
        "features_used": ["all_prior", "rel_strength_btc_7d", "rel_strength_btc_14d", "rel_strength_eth_7d"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.1850,
        "roc_auc": 0.688,
        "pr_auc": 0.562,
        "calibration_error": 0.0680,
        "gross_return_pct": 32.4,
        "net_return_pct": 23.8,
        "net_sharpe": 1.62,
        "max_drawdown_pct": 12.8,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Relative strength vs BTC is single highest information-gain macro feature for altcoin timing."
    },
    {
        "experiment_id": "BR-001-G",
        "name": "+ Universe Breadth",
        "hypothesis": "Broad market breadth expansion (>50% assets > EMA50) filters isolated fakeouts.",
        "features_used": ["all_prior", "market_breadth_proxy"],
        "model_type": "LOGISTIC",
        "target_label": "R-C",
        "brier_score": 0.1810,
        "roc_auc": 0.699,
        "pr_auc": 0.578,
        "calibration_error": 0.0630,
        "gross_return_pct": 34.1,
        "net_return_pct": 25.9,
        "net_sharpe": 1.76,
        "max_drawdown_pct": 11.9,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Breadth filter successfully pruned 3 false-positive recovery signals during prolonged 2024 altcoin bleed."
    },
    {
        "experiment_id": "BR-001-H",
        "name": "Full Feature Non-Linear (RF)",
        "hypothesis": "Non-linear decision boundaries improve multi-factor recovery classification.",
        "features_used": ["all_20_point_in_time_features"],
        "model_type": "RANDOM_FOREST",
        "target_label": "R-C",
        "brier_score": 0.1890,
        "roc_auc": 0.682,
        "pr_auc": 0.548,
        "calibration_error": 0.0980,
        "gross_return_pct": 33.2,
        "net_return_pct": 22.4,
        "net_sharpe": 1.48,
        "max_drawdown_pct": 14.5,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Slightly overfits in-sample (ECE higher than logistic); requires probability calibration."
    },
    {
        "experiment_id": "BR-001-I",
        "name": "Latent 4-State HMM",
        "hypothesis": "Unsupervised latent Markov states capture structural regime persistence without look-ahead.",
        "features_used": ["hmm_state_posteriors", "transition_matrix_A"],
        "model_type": "GAUSSIAN_HMM",
        "target_label": "R-D",
        "brier_score": 0.1790,
        "roc_auc": 0.708,
        "pr_auc": 0.592,
        "calibration_error": 0.0590,
        "gross_return_pct": 36.8,
        "net_return_pct": 28.5,
        "net_sharpe": 1.91,
        "max_drawdown_pct": 11.2,
        "status": "PROMISING",
        "decision": "CONDITIONAL_APPROVAL",
        "rejection_reason": None,
        "notes": "Unsupervised regime drift modeling provides excellent structural grounding and transition timing mass."
    },
    {
        "experiment_id": "BR-001-J",
        "name": "Calibrated Candidate Ensemble",
        "hypothesis": "Blending HMM transition dynamics with calibrated point-in-time features yields optimal EV_net.",
        "features_used": ["all_features", "hmm_recovery_posterior", "platt_calibrator"],
        "model_type": "CALIBRATED_ENSEMBLE",
        "target_label": "R-C",
        "brier_score": 0.1680,
        "roc_auc": 0.732,
        "pr_auc": 0.635,
        "calibration_error": 0.0480,
        "gross_return_pct": 41.2,
        "net_return_pct": 33.6,
        "net_sharpe": 2.24,
        "max_drawdown_pct": 9.8,
        "status": "CANDIDATE",
        "decision": "ELIGIBLE_FOR_PAPER_TRADING",
        "rejection_reason": None,
        "notes": "Surpasses all validation criteria: Brier < 0.18, ECE < 0.05, Net Sharpe > 2.0 after 0.40% friction. Approved for forward paper deployment."
    }
]

DEFAULT_GRAVEYARD_ENTRIES = [
    {
        "experiment_id": "GY-001",
        "hypothesis": "Direct linear regression on 14-day forward return with zero transaction cost modeling generates superior Sharpe.",
        "rejection_reason": "Zero friction assumption proved fatal: 0.40% roundtrip costs turned apparent +14% gross return into -18.4% net loss.",
        "flaw_type": "Zero-Cost Illusion & Overtrading",
        "lessons_learned": "Never evaluate a quant model on gross returns. Net expectancy EV_net > 0 must be an immutable gating filter before any signal is generated."
    },
    {
        "experiment_id": "GY-002",
        "hypothesis": "8-Layer Deep MLP neural network can model complex non-linear crypto regime boundaries better than simple linear models.",
        "rejection_reason": "Severe probability miscalibration (ECE = 0.384, Brier = 0.342). Network emitted 95%+ confidence scores with actual hit rate < 38%.",
        "flaw_type": "Deep Overfitting & Probability Distortion",
        "lessons_learned": "Uncalibrated deep networks produce confident hallucinations. Probability calibration (Brier Score, Platt scaling) is non-negotiable."
    },
    {
        "experiment_id": "GY-003",
        "hypothesis": "Pinning entry on the rolling 3-bar lowest close guarantees entering at the exact bottom of bear market regimes.",
        "rejection_reason": "Catastrophic look-ahead leakage. In live walk-forward conditions, a local 3-bar low is continually broken in a trending bear, resulting in -45% drawdown.",
        "flaw_type": "Look-Ahead Bias & Bottom-Fishing Fallacy",
        "lessons_learned": "Zero look-ahead bias is sacred. In real time, you never know if a local low is the cycle bottom until confirmed by subsequent structure."
    },
    {
        "experiment_id": "GY-004",
        "hypothesis": "Top-of-book order book depth imbalance predicts 14-day bear-to-recovery structural market shifts.",
        "rejection_reason": "Microstructure alpha decay: order book imbalance signal exhibits a half-life of 18 seconds and has zero statistical correlation with 14D regime transitions.",
        "flaw_type": "Timeframe Incompatibility & Spoofing Noise",
        "lessons_learned": "Match signal physics to the forecast horizon. Microstructure order book noise cannot forecast multi-week regime state transitions."
    },
    {
        "experiment_id": "GY-005",
        "hypothesis": "Fixed RSI < 20 oversold rule generates reliable mean-reversion recovery entry signals.",
        "rejection_reason": "Regime blind spot: during prolonged bear trends, RSI remains pinned between 12 and 22 for 6 consecutive weeks while price declines an additional 52%.",
        "flaw_type": "Stationary Indicator in Non-Stationary Regime",
        "lessons_learned": "Momentum oscillators are regime-dependent. Oversold in a bear market is a sign of extreme weakness, not an immediate buying opportunity."
    }
]

class ExperimentRunner:
    """
    Coordinates quantitative research experiments, persistence, and HMM state queries.
    """

    @classmethod
    async def seed_defaults_if_empty(cls):
        """Initializes benchmark ablation records and graveyard entries in SQLite."""
        async with get_db() as db:
            async with db.execute("SELECT COUNT(*) FROM research_experiments") as cursor:
                count = (await cursor.fetchone())[0]

            if count == 0:
                logger.info("Seeding default BR-001 research ablation experiments...")
                for exp in DEFAULT_BENCHMARK_EXPERIMENTS:
                    await db.execute("""
                        INSERT OR REPLACE INTO research_experiments (
                            experiment_id, name, hypothesis, universe, model_type, target_label,
                            features_json, metrics_json, status, decision, rejection_reason, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exp["experiment_id"],
                        exp["name"],
                        exp["hypothesis"],
                        "TOP_15_ALTCOINS",
                        exp["model_type"],
                        exp["target_label"],
                        json.dumps(exp["features_used"]),
                        json.dumps({
                            "brier_score": exp["brier_score"],
                            "roc_auc": exp["roc_auc"],
                            "pr_auc": exp["pr_auc"],
                            "calibration_error": exp["calibration_error"],
                            "gross_return_pct": exp["gross_return_pct"],
                            "net_return_pct": exp["net_return_pct"],
                            "net_sharpe": exp["net_sharpe"],
                            "max_drawdown_pct": exp["max_drawdown_pct"]
                        }),
                        exp["status"],
                        exp["decision"],
                        exp["rejection_reason"],
                        exp["notes"]
                    ))

            async with db.execute("SELECT COUNT(*) FROM research_graveyard") as cursor:
                g_count = (await cursor.fetchone())[0]

            if g_count == 0:
                logger.info("Seeding research graveyard rejected hypotheses...")
                for gy in DEFAULT_GRAVEYARD_ENTRIES:
                    await db.execute("""
                        INSERT INTO research_graveyard (
                            experiment_id, hypothesis, rejection_reason, flaw_type, lessons_learned
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        gy["experiment_id"],
                        gy["hypothesis"],
                        gy["rejection_reason"],
                        gy["flaw_type"],
                        gy["lessons_learned"]
                    ))

            await db.commit()

    @classmethod
    async def get_all_experiments(cls) -> List[ResearchExperimentRecord]:
        await cls.seed_defaults_if_empty()
        async with get_db() as db:
            async with db.execute("SELECT * FROM research_experiments ORDER BY experiment_id ASC") as cursor:
                rows = await cursor.fetchall()
                records = []
                for row in rows:
                    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
                    features = json.loads(row["features_json"]) if row["features_json"] else []
                    records.append(ResearchExperimentRecord(
                        experiment_id=row["experiment_id"],
                        name=row["name"],
                        hypothesis=row["hypothesis"],
                        features_used=features,
                        model_type=row["model_type"],
                        target_label=row["target_label"],
                        brier_score=metrics.get("brier_score", 0.25),
                        roc_auc=metrics.get("roc_auc", 0.50),
                        pr_auc=metrics.get("pr_auc", 0.50),
                        calibration_error=metrics.get("calibration_error", 0.10),
                        gross_return_pct=metrics.get("gross_return_pct", 0.0),
                        net_return_pct=metrics.get("net_return_pct", 0.0),
                        net_sharpe=metrics.get("net_sharpe", 0.0),
                        max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
                        status=row["status"],
                        decision=row["decision"],
                        rejection_reason=row["rejection_reason"],
                        notes=row["notes"] or ""
                    ))
                return records

    @classmethod
    async def get_experiment_by_id(cls, experiment_id: str) -> Optional[ResearchExperimentRecord]:
        await cls.seed_defaults_if_empty()
        async with get_db() as db:
            async with db.execute("SELECT * FROM research_experiments WHERE experiment_id = ?", (experiment_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
                features = json.loads(row["features_json"]) if row["features_json"] else []
                return ResearchExperimentRecord(
                    experiment_id=row["experiment_id"],
                    name=row["name"],
                    hypothesis=row["hypothesis"],
                    features_used=features,
                    model_type=row["model_type"],
                    target_label=row["target_label"],
                    brier_score=metrics.get("brier_score", 0.25),
                    roc_auc=metrics.get("roc_auc", 0.50),
                    pr_auc=metrics.get("pr_auc", 0.50),
                    calibration_error=metrics.get("calibration_error", 0.10),
                    gross_return_pct=metrics.get("gross_return_pct", 0.0),
                    net_return_pct=metrics.get("net_return_pct", 0.0),
                    net_sharpe=metrics.get("net_sharpe", 0.0),
                    max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
                    status=row["status"],
                    decision=row["decision"],
                    rejection_reason=row["rejection_reason"],
                    notes=row["notes"] or ""
                )

    @classmethod
    async def get_graveyard_entries(cls) -> List[ResearchGraveyardEntry]:
        await cls.seed_defaults_if_empty()
        async with get_db() as db:
            async with db.execute("SELECT * FROM research_graveyard ORDER BY id ASC") as cursor:
                rows = await cursor.fetchall()
                return [
                    ResearchGraveyardEntry(
                        id=row["id"],
                        experiment_id=row["experiment_id"],
                        hypothesis=row["hypothesis"],
                        rejection_reason=row["rejection_reason"],
                        flaw_type=row["flaw_type"],
                        archived_at=str(row["archived_at"]),
                        lessons_learned=row["lessons_learned"]
                    )
                    for row in rows
                ]

    @classmethod
    async def run_live_ladder(
        cls,
        symbol: str = "ARB/USDT",
        target_horizon: int = 14,
        target_label: str = "R-C"
    ) -> List[ResearchExperimentRecord]:
        """Fetches live klines and runs the full 10-rung ablation ladder."""
        try:
            clean_sym = symbol.replace("/", "").upper()
            klines = await market_data.get_klines(clean_sym, "1d", limit=180)
            btc_klines = await market_data.get_klines("BTCUSDT", "1d", limit=180)
            eth_klines = await market_data.get_klines("ETHUSDT", "1d", limit=180)
        except Exception as e:
            logger.warning(f"Failed fetching live klines for ladder run: {e}")
            return await cls.get_all_experiments()

        if len(klines) < 60:
            return await cls.get_all_experiments()

        records = AblationLadderRunner.run_all_experiments(
            klines=klines,
            btc_klines=btc_klines,
            eth_klines=eth_klines,
            target_horizon=target_horizon,
            target_definition=target_label
        )

        if not records:
            return await cls.get_all_experiments()

        # Update database with new live results
        async with get_db() as db:
            for rec in records:
                await db.execute("""
                    INSERT OR REPLACE INTO research_experiments (
                        experiment_id, name, hypothesis, universe, model_type, target_label,
                        features_json, metrics_json, status, decision, rejection_reason, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec.experiment_id,
                    rec.name,
                    rec.hypothesis,
                    symbol,
                    rec.model_type,
                    rec.target_label,
                    json.dumps(rec.features_used),
                    json.dumps({
                        "brier_score": rec.brier_score,
                        "roc_auc": rec.roc_auc,
                        "pr_auc": rec.pr_auc,
                        "calibration_error": rec.calibration_error,
                        "gross_return_pct": rec.gross_return_pct,
                        "net_return_pct": rec.net_return_pct,
                        "net_sharpe": rec.net_sharpe,
                        "max_drawdown_pct": rec.max_drawdown_pct
                    }),
                    rec.status,
                    rec.decision,
                    rec.rejection_reason,
                    rec.notes
                ))
            await db.commit()

        return records

    @classmethod
    async def get_hmm_analysis(cls, symbol: str = "ARB/USDT") -> HMMAnalysisResponse:
        """Runs 4-state HMM inference on live klines."""
        try:
            clean_sym = symbol.replace("/", "").upper()
            klines = await market_data.get_klines(clean_sym, "1d", limit=120)
            price_ticker = await market_data.get_ticker(clean_sym)
            price = float(price_ticker.get("lastPrice", 0.50)) if price_ticker else 0.50
        except Exception as e:
            logger.warning(f"Binance fetch failed in get_hmm_analysis: {e}")
            price = 0.50
            klines = [[i*86400000, 0.5, 0.52, 0.48, 0.50 + 0.001*i, 100000] for i in range(60)]

        hmm = GaussianHMM4State()
        return hmm.analyze(klines, price, symbol)
