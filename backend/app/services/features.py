import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class FeatureEngine:
    @staticmethod
    def calculate_indicators(klines: List[List[Any]]) -> Optional[Dict[str, Any]]:
        """
        Calculates full quantitative indicator stack from klines:
        klines: [[timestamp, open, high, low, close, volume], ...]
        """
        if not klines or len(klines) < 35:
            return None

        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # 1. EMAs: 20, 50, 200
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean() if len(df) >= 50 else df["ema_20"]
        df["ema_200"] = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else df["ema_50"]
        
        # EMA 20 slope (% change over 3 bars)
        ema_20_val = df["ema_20"].iloc[-1]
        ema_20_prev = df["ema_20"].iloc[-4] if len(df) >= 4 else ema_20_val
        ema_slope_20 = ((ema_20_val - ema_20_prev) / ema_20_prev) * 100.0 if ema_20_prev > 0 else 0.0

        # 2. RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 3. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # 4. ATR (14)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        if np.isnan(atr_14):
            atr_14 = tr.mean()

        # 5. Bollinger Bands (20, 2)
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-9)

        # 6. ADX (14)
        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr.rolling(14).sum()
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(14).mean().iloc[-1]
        if np.isnan(adx):
            adx = 20.0

        # 7. Volume & RVOL (20 period)
        vol_sma_20 = volume.rolling(20).mean()
        current_vol = volume.iloc[-1]
        rvol = current_vol / (vol_sma_20.iloc[-1] + 1e-9)
        prev_rvol = volume.iloc[-2] / (vol_sma_20.iloc[-2] + 1e-9) if len(volume) >= 2 else rvol
        vol_accel = rvol - prev_rvol

        # 8. On-Balance Volume (OBV)
        obv_direction = np.where(close > close.shift(), 1, np.where(close < close.shift(), -1, 0))
        obv = (volume * obv_direction).cumsum().iloc[-1]

        current_price = float(close.iloc[-1])

        return {
            "price": current_price,
            "ema_20": float(df["ema_20"].iloc[-1]),
            "ema_50": float(df["ema_50"].iloc[-1]),
            "ema_200": float(df["ema_200"].iloc[-1]),
            "ema_slope_20": float(ema_slope_20),
            "adx_14": float(adx),
            "plus_di": float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0.0,
            "minus_di": float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0.0,
            "rsi_14": float(df["rsi"].iloc[-1]),
            "macd_line": float(macd_line.iloc[-1]),
            "macd_signal": float(macd_signal.iloc[-1]),
            "macd_hist": float(macd_hist.iloc[-1]),
            "atr_14": float(atr_14),
            "bb_upper": float(bb_upper.iloc[-1]),
            "bb_middle": float(bb_middle.iloc[-1]),
            "bb_lower": float(bb_lower.iloc[-1]),
            "bb_width": float(bb_width.iloc[-1]),
            "rvol_20": float(rvol),
            "volume_accel": float(vol_accel),
            "obv": float(obv)
        }

feature_engine = FeatureEngine()
