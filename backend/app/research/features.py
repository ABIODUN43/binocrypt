import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class FeatureExtractor:
    """
    Point-in-Time Quantitative Feature Extraction Library for BR-001.
    Strictly guarantees zero look-ahead bias:
    - All rolling calculations use only observations at or before time t: X_t = f(P_{<=t}, V_{<=t}).
    - Forward returns and targets are strictly segregated.
    """

    FEATURE_GROUPS = {
        "trend": ["dist_ema20", "dist_ema50", "dist_ema200", "ema20_slope_5d", "ema50_slope_5d"],
        "momentum": ["rsi14", "macd_hist", "macd_hist_slope_3d", "mom_7d", "mom_14d"],
        "volume": ["rvol20", "vol_trend_5d", "vol_price_divergence"],
        "volatility": ["realized_vol_14d", "atr_pct_14d", "bb_width"],
        "drawdown_liquidity": ["dd_from_30d_high", "dd_from_90d_high", "amihud_illiquidity_proxy"],
        "macro_context": ["rel_strength_btc_7d", "rel_strength_btc_14d", "rel_strength_eth_7d"],
        "breadth": ["market_breadth_proxy"]
    }

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    @classmethod
    def extract_features(
        cls,
        klines: List[List[Any]],
        btc_klines: Optional[List[List[Any]]] = None,
        eth_klines: Optional[List[List[Any]]] = None,
        breadth_ratio: float = 0.50
    ) -> pd.DataFrame:
        """
        Extracts full set of 20+ quantitative features from historical klines.
        klines format: [timestamp, open, high, low, close, volume, ...]
        """
        if not klines or len(klines) < 30:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ][:len(klines[0])])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        feat = pd.DataFrame(index=df.index)
        feat["timestamp"] = df["timestamp"]
        feat["close"] = close

        # --- 1. Trend Group ---
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        feat["dist_ema20"] = (close - ema20) / (ema20 + 1e-9)
        feat["dist_ema50"] = (close - ema50) / (ema50 + 1e-9)
        feat["dist_ema200"] = (close - ema200) / (ema200 + 1e-9)
        feat["ema20_slope_5d"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)
        feat["ema50_slope_5d"] = (ema50 - ema50.shift(5)) / (ema50.shift(5) + 1e-9)

        # --- 2. Momentum Group ---
        feat["rsi14"] = cls._compute_rsi(close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        feat["macd_hist"] = macd_hist / (close + 1e-9)
        feat["macd_hist_slope_3d"] = (macd_hist - macd_hist.shift(3)) / (close + 1e-9)
        feat["mom_7d"] = (close - close.shift(7)) / (close.shift(7) + 1e-9)
        feat["mom_14d"] = (close - close.shift(14)) / (close.shift(14) + 1e-9)

        # --- 3. Volume Group ---
        vol_sma20 = vol.rolling(window=20, min_periods=5).mean()
        feat["rvol20"] = vol / (vol_sma20 + 1e-9)
        vol_sma5 = vol.rolling(window=5, min_periods=2).mean()
        feat["vol_trend_5d"] = (vol_sma5 - vol_sma20) / (vol_sma20 + 1e-9)
        ret_3d = (close - close.shift(3)) / (close.shift(3) + 1e-9)
        vol_chg_3d = (vol - vol.shift(3)) / (vol.shift(3) + 1e-9)
        feat["vol_price_divergence"] = np.sign(ret_3d) * -1.0 * np.sign(vol_chg_3d)

        # --- 4. Volatility Group ---
        log_ret = np.log(close / (close.shift(1) + 1e-9) + 1e-9)
        feat["realized_vol_14d"] = log_ret.rolling(window=14, min_periods=7).std() * np.sqrt(365)
        atr14 = cls._compute_atr(high, low, close, 14)
        feat["atr_pct_14d"] = atr14 / (close + 1e-9)
        sma20 = close.rolling(window=20, min_periods=10).mean()
        std20 = close.rolling(window=20, min_periods=10).std()
        feat["bb_width"] = (4.0 * std20) / (sma20 + 1e-9)

        # --- 5. Drawdown & Liquidity Proxy ---
        rolling_max_30 = high.rolling(window=30, min_periods=10).max()
        rolling_max_90 = high.rolling(window=90, min_periods=20).max()
        feat["dd_from_30d_high"] = (close - rolling_max_30) / (rolling_max_30 + 1e-9)
        feat["dd_from_90d_high"] = (close - rolling_max_90) / (rolling_max_90 + 1e-9)
        dollar_vol = close * vol
        feat["amihud_illiquidity_proxy"] = (log_ret.abs() / (dollar_vol + 1e-5)).rolling(14).mean()

        # --- 6. Macro & Relative Context ---
        if btc_klines and len(btc_klines) >= len(klines):
            btc_closes = pd.Series([float(k[4]) for k in btc_klines[-len(klines):]], index=df.index)
            btc_ret_7d = (btc_closes - btc_closes.shift(7)) / (btc_closes.shift(7) + 1e-9)
            btc_ret_14d = (btc_closes - btc_closes.shift(14)) / (btc_closes.shift(14) + 1e-9)
            feat["rel_strength_btc_7d"] = feat["mom_7d"] - btc_ret_7d
            feat["rel_strength_btc_14d"] = feat["mom_14d"] - btc_ret_14d
        else:
            feat["rel_strength_btc_7d"] = feat["mom_7d"] * 0.5
            feat["rel_strength_btc_14d"] = feat["mom_14d"] * 0.5

        if eth_klines and len(eth_klines) >= len(klines):
            eth_closes = pd.Series([float(k[4]) for k in eth_klines[-len(klines):]], index=df.index)
            eth_ret_7d = (eth_closes - eth_closes.shift(7)) / (eth_closes.shift(7) + 1e-9)
            feat["rel_strength_eth_7d"] = feat["mom_7d"] - eth_ret_7d
        else:
            feat["rel_strength_eth_7d"] = feat["mom_7d"] * 0.4

        # --- 7. Breadth Proxy ---
        feat["market_breadth_proxy"] = float(breadth_ratio)

        # Clean fill
        numeric_cols = [c for c in feat.columns if c not in ["timestamp", "close"]]
        feat[numeric_cols] = feat[numeric_cols].ffill().fillna(0.0)

        return feat

    @classmethod
    def get_feature_subset(cls, df: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
        """Returns feature columns corresponding to specified feature groups."""
        selected_cols = []
        for g in groups:
            if g in cls.FEATURE_GROUPS:
                selected_cols.extend(cls.FEATURE_GROUPS[g])
        valid_cols = [c for c in selected_cols if c in df.columns]
        return df[valid_cols]
