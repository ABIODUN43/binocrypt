import pytest
import numpy as np
import asyncio
from app.services.features import feature_engine
from app.services.market_structure import market_structure_analyzer
from app.services.scoring import scoring_engine
from app.services.risk_engine import risk_engine
from app.services.backtester import backtester

def generate_mock_klines(n=100, base_price=100.0, trend=0.1):
    """Generates synthetic OHLCV bars for unit testing."""
    klines = []
    price = base_price
    for i in range(n):
        timestamp = 1700000000000 + (i * 3600 * 1000)
        open_p = price
        change = (np.sin(i / 5.0) * 1.5) + (trend * i)
        high_p = open_p + abs(change) + 0.8
        low_p = max(1.0, open_p - abs(change) - 0.5)
        close_p = (open_p + high_p + low_p) / 3.0
        volume = 1000.0 + (i * 10.0) + (np.sin(i) * 300.0)
        price = close_p
        klines.append([timestamp, open_p, high_p, low_p, close_p, volume])
    return klines

def test_feature_engine():
    klines = generate_mock_klines(n=60, base_price=100.0)
    feat = feature_engine.calculate_indicators(klines)
    assert feat is not None
    assert "ema_20" in feat
    assert "rsi_14" in feat
    assert 0 <= feat["rsi_14"] <= 100
    assert feat["atr_14"] > 0
    assert feat["rvol_20"] > 0
    assert "macd_hist" in feat

def test_market_structure():
    klines = generate_mock_klines(n=50, base_price=100.0, trend=0.3)
    struct = market_structure_analyzer.analyze_structure(klines)
    assert "structure_type" in struct
    assert struct["support_level"] > 0
    assert struct["recent_swing_low"] > 0

def test_opportunity_scoring():
    klines = generate_mock_klines(n=50, base_price=100.0)
    feat = feature_engine.calculate_indicators(klines)
    struct = market_structure_analyzer.analyze_structure(klines)
    ticker = {
        "volume_24h_usd": 25_000_000,
        "change_24h": 3.5,
        "spread_pct": 0.04
    }
    score = scoring_engine.score_candidate(
        symbol="TESTUSDT",
        ticker=ticker,
        features_1h=feat,
        features_4h=None,
        structure_1h=struct,
        regime="BULL_TREND"
    )
    assert 0.0 <= score.total_score <= 100.0
    assert score.grade in ["EXCELLENT", "STRONG", "GOOD", "WATCH", "IGNORE"]
    assert 0.0 <= score.win_probability <= 1.0

def test_risk_engine_small_account():
    # Account = $40, Risk = 2% ($0.80), Entry = $100, Stop = $96 (4% stop distance)
    # Expected Position Size = $0.80 / 0.04 = $20.00
    res = risk_engine.calculate_position_size(
        account_equity=40.0,
        risk_pct=2.0,
        entry_price=100.0,
        stop_loss=96.0
    )
    assert res.is_valid is True
    assert res.risk_amount_usd == 0.80
    assert res.stop_distance_pct == 4.0
    assert res.recommended_position_usd == 20.0
    assert res.recommended_quantity == 0.2

def test_backtester_simulation():
    klines = generate_mock_klines(n=120, base_price=100.0, trend=0.2)
    res = backtester.run_backtest(
        klines=klines,
        symbol="BTCUSDT",
        timeframe="1h",
        initial_capital=40.0,
        risk_pct=2.0
    )
    assert res.starting_capital == 40.0
    assert res.total_bars == 120
    assert isinstance(res.total_trades, int)
    assert isinstance(res.trades, list)
