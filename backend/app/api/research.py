import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from ..models.schemas import (
    ResearchExperimentRecord,
    ResearchGraveyardEntry,
    HMMAnalysisResponse
)
from ..research.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Quantitative Research"])

@router.get("/experiments", response_model=List[ResearchExperimentRecord])
async def list_experiments():
    """
    Returns the reproducible ablation ladder experiments BR-001-A through BR-001-J,
    including in-sample/out-sample Brier scores, calibration errors, and net performance metrics.
    """
    try:
        experiments = await ExperimentRunner.get_all_experiments()
        return experiments
    except Exception as e:
        logger.error(f"Failed to fetch research experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}", response_model=ResearchExperimentRecord)
async def get_experiment(experiment_id: str):
    """
    Returns deep-dive report for a specific research ablation experiment by ID.
    """
    try:
        record = await ExperimentRunner.get_experiment_by_id(experiment_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
        return record
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-ladder", response_model=List[ResearchExperimentRecord])
async def run_ablation_ladder(
    symbol: str = Query("ARB/USDT", description="Crypto trading pair to evaluate"),
    target_horizon: int = Query(14, ge=7, le=30, description="Forward forecast horizon in days"),
    target_label: str = Query("R-C", description="Target definition: R-A, R-B, R-C, R-D")
):
    """
    Triggers an on-demand re-evaluation of the 10-rung ablation ladder on live market data.
    """
    try:
        clean_symbol = symbol.replace("-", "/").upper()
        if not clean_symbol.endswith("/USDT") and not clean_symbol.endswith("USDT"):
            clean_symbol = f"{clean_symbol}/USDT"
        records = await ExperimentRunner.run_live_ladder(
            symbol=clean_symbol,
            target_horizon=target_horizon,
            target_label=target_label
        )
        return records
    except Exception as e:
        logger.error(f"Error executing ablation ladder for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graveyard", response_model=List[ResearchGraveyardEntry])
async def list_graveyard():
    """
    Returns the Research Graveyard of rejected hypotheses, flaw types,
    and lessons learned to prevent repeat mistakes and look-ahead illusions.
    """
    try:
        entries = await ExperimentRunner.get_graveyard_entries()
        return entries
    except Exception as e:
        logger.error(f"Failed to fetch research graveyard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hmm/{symbol:path}", response_model=HMMAnalysisResponse)
async def get_hmm_state(symbol: str = "ARB/USDT"):
    """
    Returns the 4-state latent Hidden Markov Model analysis for the specified asset:
    - Current state posteriors (Bear, Sideways, Recovery, Bull)
    - Empirical transition matrix A_ij
    - 30-day hitting time probability mass distribution for Recovery
    - Estimated modal transition window
    """
    try:
        clean_symbol = symbol.replace("-", "/").upper()
        if not clean_symbol.endswith("/USDT") and not clean_symbol.endswith("USDT"):
            clean_symbol = f"{clean_symbol}/USDT"
        response = await ExperimentRunner.get_hmm_analysis(clean_symbol)
        return response
    except Exception as e:
        logger.error(f"Failed to compute HMM analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/large-move/{symbol:path}")
async def get_large_move_surface(symbol: str = "ARB/USDT"):
    """
    BR-001.1: Returns the conditional opportunity surface O(h,r) and downside risk surface D(h,d).
    """
    try:
        from ..research.large_move_engine import LargeMoveEngine
        from ..services.market_data import market_data
        clean_sym = symbol.replace("/", "").upper()
        klines = await market_data.get_klines(clean_sym, "1d", limit=120)
        btc_klines = await market_data.get_klines("BTCUSDT", "1d", limit=120)
        eth_klines = await market_data.get_klines("ETHUSDT", "1d", limit=120)
        return LargeMoveEngine.compute_opportunity_surface(klines, btc_klines, eth_klines)
    except Exception as e:
        logger.error(f"Failed to compute large move surface for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entry-strategies/{symbol:path}")
async def get_entry_strategies(symbol: str = "ARB/USDT"):
    """
    BR-001.2: Returns comparative performance for E1 (Early), E2 (Confirmed), E3 (Combined), E4 (Cost-Aware).
    """
    try:
        from ..research.entry_strategy_eval import EntryStrategyEvaluator
        from ..services.market_data import market_data
        clean_sym = symbol.replace("/", "").upper()
        klines = await market_data.get_klines(clean_sym, "1d", limit=120)
        btc_klines = await market_data.get_klines("BTCUSDT", "1d", limit=120)
        eth_klines = await market_data.get_klines("ETHUSDT", "1d", limit=120)
        return EntryStrategyEvaluator.evaluate_entry_strategies(klines, btc_klines, eth_klines)
    except Exception as e:
        logger.error(f"Failed to evaluate entry strategies for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/false-recovery/{symbol:path}")
async def get_false_recovery_risk(symbol: str = "ARB/USDT"):
    """
    BR-001.3: Returns probability of false recovery / bull trap vs. transition maturation.
    """
    try:
        from ..research.false_recovery_model import FalseRecoveryModel
        from ..services.market_data import market_data
        clean_sym = symbol.replace("/", "").upper()
        klines = await market_data.get_klines(clean_sym, "1d", limit=120)
        return FalseRecoveryModel.evaluate_false_recovery_risk(klines, symbol, 14)
    except Exception as e:
        logger.error(f"Failed to evaluate false recovery for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ranking")
async def get_universe_ranking(limit: int = Query(10, ge=3, le=20)):
    """
    BR-001.4: Returns liquid cryptocurrency universe ranked by composite risk-adjusted opportunity score.
    """
    try:
        from ..research.ranking_engine import CrossAssetRankingEngine
        return await CrossAssetRankingEngine.rank_universe(limit=limit)
    except Exception as e:
        logger.error(f"Failed to rank universe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

