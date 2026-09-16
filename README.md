# Binocrypt — Quantitative Crypto Market Intelligence & Decision Support

> **Spot-Only Trading System** engineered for continuous liquid universe scanning, multi-timeframe regime detection, transparent opportunity scoring, and small-account capital preservation (The $40 Experiment).

---

## 1. Core Philosophy & System Architecture

Binocrypt is **not** a "crypto signal bot" and does not instruct AI to "make 5x". It is a quantitative decision-support system where:
- **The Quantitative Engine calculates mathematical evidence** (multi-timeframe indicators, market structure pivots, regime alignment, risk/reward).
- **The Risk Engine sizes positions strictly from stop-loss distance** to safeguard small accounts ($40 capital).
- **The AI Synthesis Layer translates structured evidence into human-readable explanations** without hallucinating numbers or trade signals.
- **Spot Trading Only**: All leverage, futures, perpetuals, margin, and shorting mechanics are strictly excluded. Directional states are: `LONG`, `HOLD`, `SELL`, and `NO TRADE`.

```
                 BINOCRYPT ARCHITECTURE
                     │
                     ▼
          Binance Spot Market Data
                     │
                     ▼
            Liquid Spot Universe
        (Vol > $5M, Spread < 0.20%)
                     │
                     ▼
          Multi-Timeframe Analysis
          (4H Macro | 1H Trend | 15M Entry)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Market Regime          Coin Features
   (BTC/ETH Trend,        (EMA, RSI, MACD,
  Breadth, Volatility)    RVOL, Structure)
          │                     │
          └──────────┬──────────┘
                     ▼
         Opportunity Score (0–100)
                     │
                     ▼
            Trade Setup Engine
        (Entry, Stop, TP1, TP2, R:R)
                     │
                     ▼
                Risk Engine
     (Position Size = Risk / StopDistance)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Paper Trading          Backtest Lab
   ($40 Experiment)     (Historical Replay)
```

---

## 2. Key Modules

### Market Regime Engine
Classifies the global crypto environment using BTC & ETH multi-timeframe trends (4H & 1H), altcoin market breadth (% of universe trading above their 50 EMA), and normalized ATR volatility:
- `BULL_TREND`: Aggressive spot trend continuation setups permitted.
- `BEAR_TREND`: Cash defense active; no new long entries.
- `SIDEWAYS`: Selective range and support bounces only.
- `HIGH_VOLATILITY / PANIC`: Trading halted to avoid slippage and violent whipsaws.

### 100-Point Transparent Opportunity Scoring
- **Trend Confluence (20 pts)**: EMA 20/50/200 stacking, positive slope, ADX > 22.
- **Momentum Engine (20 pts)**: RSI sweet spot (52–68), positive & expanding MACD histogram.
- **Volume Influx (20 pts)**: RVOL >= 1.3x over 20-period volume SMA, volume acceleration.
- **Market Structure (20 pts)**: Confirmed Higher Highs/Lows, Breakout, or Pullback to support.
- **Liquidity & Spread (10 pts)**: Spread <= 0.08%, 24h USD volume > $15M.
- **Volatility (10 pts)**: Bollinger Band expansion from squeeze, stable ATR.
- **Regime Alignment (10 pts)**: Confluence with global BTC/ETH regime.
- **Risk/Reward (10 pts)**: Setup R:R >= 1.8.

**Tiers**:
- 90–100: `EXCELLENT`
- 80–89: `STRONG`
- 70–79: `GOOD`
- 60–69: `WATCH`
- <60: `IGNORE`

### Risk Engine & The $40 Experiment
For a $40 account targeting $150–$200:
$$\text{Position Size (\$) } = \frac{\text{Risk Amount (\$) }}{\text{Stop Distance Ratio}}$$
- Account Equity: \$40.00
- Risk per Trade: strictly 1.5%–2.0% (\$0.60–\$0.80).
- Max Open Positions: 2 positions.
- Daily Drawdown Circuit Breaker: 5% (\$2.00). If hit, halts trading for the day.

---

## 3. Quick Start

### Backend (Python FastAPI)
```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation: `http://127.0.0.1:8000/docs`

### Frontend (React + Vite + Tailwind)
```bash
cd frontend
cmd.exe /c "npm run dev"
```
Terminal UI: `http://localhost:5173`

---

## 4. Testing
Run the quantitative test suite:
```bash
cd backend
python -m pytest tests
```
