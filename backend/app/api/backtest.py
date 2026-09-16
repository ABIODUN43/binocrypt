from fastapi import APIRouter, HTTPException
from ..models.schemas import BacktestRequest, BacktestResult
from ..services.market_data import market_data
from ..services.backtester import backtester
from ..core.config import settings

router = APIRouter(prefix="/backtest", tags=["Backtester"])

@router.post("/run", response_model=BacktestResult)
async def run_backtest_simulation(req: BacktestRequest):
    """
    Runs historical quantitative backtest on real Binance spot candles
    with fees, slippage, and strict bar-by-bar execution.
    """
    limit = min(500, max(100, req.limit_bars))
    klines = await market_data.get_klines(symbol=req.symbol.upper(), interval=req.timeframe, limit=limit)
    
    if not klines or len(klines) < 60:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not retrieve sufficient historical data for {req.symbol} on {req.timeframe}."
        )

    try:
        result = backtester.run_backtest(
            klines=klines,
            symbol=req.symbol.upper(),
            timeframe=req.timeframe,
            initial_capital=settings.DEFAULT_CAPITAL,
            risk_pct=req.risk_pct,
            fee_pct=settings.DEFAULT_FEE_PCT,
            slippage_pct=settings.DEFAULT_SLIPPAGE_PCT,
            target_rr=req.rr_target
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")
