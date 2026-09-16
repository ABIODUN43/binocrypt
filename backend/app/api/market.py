from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any
from ..services.market_data import market_data
from ..services.regime import regime_engine
from ..models.schemas import MarketRegime, Candle

router = APIRouter(prefix="/market", tags=["Market Data"])

@router.get("/regime", response_model=MarketRegime)
async def get_market_regime():
    """Returns current market regime model with BTC/ETH trend and breadth."""
    return await regime_engine.detect_regime()

@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Fetch 24hr ticker for symbol."""
    ticker = await market_data.get_ticker(symbol)
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Ticker not found for {symbol}")
    return ticker

@router.get("/klines")
async def get_klines(
    symbol: str = Query("BTCUSDT", description="Crypto pair e.g. SOLUSDT"),
    interval: str = Query("1h", description="15m, 1h, 4h, 1d"),
    limit: int = Query(100, ge=10, le=500)
):
    """Fetches formatted OHLCV klines."""
    raw_klines = await market_data.get_klines(symbol=symbol, interval=interval, limit=limit)
    candles = []
    for k in raw_klines:
        candles.append({
            "time": int(k[0] / 1000),  # Unix seconds for chart libraries
            "open": k[1],
            "high": k[2],
            "low": k[3],
            "close": k[4],
            "volume": k[5]
        })
    return candles

@router.get("/depth/{symbol}")
async def get_depth(symbol: str):
    """Fetches orderbook spread and liquidity depth."""
    return await market_data.get_orderbook_depth(symbol)
