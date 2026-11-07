"""
BINOCRYPT API - BR-009: Probabilistic Recovery & Optimal Entry Intelligence
Provides endpoints for:
- 14-Day Multi-Barrier Target-Specific Recovery Prediction
- Optimal Entry Region & Strategy Benchmarking (A to F)
- Universe Depressed Market Recovery Scanner
- Empirical Probability Calibration & Reliability Metrics
- Atomic Execution Safety via BR-008 Signal Guard
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from ..services.recovery_engine import recovery_engine
from ..services.signal_guard import signal_guard

router = APIRouter(prefix="/recovery", tags=["recovery-intelligence"])

class ExecuteRecoveryTradeRequest(BaseModel):
    symbol: str
    preferred_entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    signal_id: Optional[str] = None
    client_order_id: Optional[str] = None

@router.get("/scanner")
async def get_recovery_scanner():
    """
    BR-009 Universe Recovery Scanner.
    Scans liquid crypto pairs for 14-day recovery opportunities, ranked by Net Expectancy and Depression.
    """
    try:
        items = await recovery_engine.get_recovery_scanner()
        return {
            "success": True,
            "count": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan recovery universe: {str(e)}")

@router.get("/calibration")
async def get_recovery_calibration():
    """
    BR-009 Probability Calibration Diagnostics.
    Returns out-of-sample Brier score, log loss, and observed vs predicted reliability bins.
    """
    try:
        cal = recovery_engine._get_calibration_metadata(0.70)
        return {
            "success": True,
            "calibration": cal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch calibration diagnostics: {str(e)}")

@router.get("/strategies/{symbol}")
async def get_strategy_benchmarks(symbol: str):
    """
    BR-009 Entry Strategy Comparison Benchmark.
    Compares the 6 candidate entry strategies (Deep Depression, Exhaustion, Stabilisation, Early Recovery, Confirmed Reversal, Pullback).
    """
    try:
        data = await recovery_engine.get_recovery_intelligence(symbol)
        return {
            "success": True,
            "symbol": data["symbol"],
            "current_price": data["current_price"],
            "benchmarks": data.get("entry_strategy_benchmarks", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategy benchmarks for {symbol}: {str(e)}")

@router.get("/{symbol}")
async def get_recovery_intelligence(symbol: str):
    """
    BR-009 Complete Probabilistic Recovery & Optimal Entry Intelligence Report.
    Returns 14-day target-specific recovery matrix (TP1 to TP4), optimal entry region, and decision.
    """
    try:
        report = await recovery_engine.get_recovery_intelligence(symbol)
        return {
            "success": True,
            "data": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute recovery intelligence for {symbol}: {str(e)}")

@router.post("/execute")
async def execute_recovery_trade(req: ExecuteRecoveryTradeRequest):
    """
    Atomic Execution of a BR-009 Recommended Trade.
    Routes through BR-008 SignalLifecycleGuard to enforce data freshness, price-runaway protection, and duplicate prevention.
    """
    try:
        res = await signal_guard.execute_signal(
            symbol=req.symbol,
            signal_id=req.signal_id,
            client_order_id=req.client_order_id
        )
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Execution rejected by safety guard."))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
