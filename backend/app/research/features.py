import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

class FeatureExtractor:
    """
    Point-in-Time Quantitative Feature Extraction Library for BR-001 / BR-002 / BR-003.

    Strictly guarantees zero look-ahead bias:
    - All rolling calculations use only observations at or before time t.
    - X_t = f(P_{<=t}, V_{<=t}, High_{<=t}, Low_{<=t}).
    - Forward returns and labels are strictly segregated into a separate file.

    ATH/ATL are computed as expanding-window maxima/minima (causal).
    Rolling extremes (30D/90D/180D/365D) are retained alongside expanding
    ATH/ATL because expanding values become weakly informative over long histories.
    """

    FEATURE_GROUPS = {
        # ── Existing BR-001 groups ──────────────────────────────────────────
        "trend": [
            "dist_ema20", "dist_ema50", "dist_ema200",
            "ema20_slope_5d", "ema50_slope_5d",
        ],
        "momentum": [
            "rsi14", "macd_hist", "macd_hist_slope_3d", "mom_7d", "mom_14d",
        ],
        "volume": ["rvol20", "vol_trend_5d", "vol_price_divergence"],
        "volatility": ["realized_vol_14d", "atr_pct_14d", "bb_width"],
        "drawdown_liquidity": [
            "dd_from_30d_high", "dd_from_90d_high", "amihud_illiquidity_proxy",
        ],
        "macro_context": [
            "rel_strength_btc_7d", "rel_strength_btc_14d", "rel_strength_eth_7d",
        ],
        "breadth": ["market_breadth_proxy"],

        # ── New BR-002 groups ───────────────────────────────────────────────

        # Rolling distance-to-extreme (primary signal for current cycle position)
        "drawdown_rolling": [
            "dd_from_30d_high",   "dd_from_90d_high",
            "dd_from_180d_high",  "dd_from_365d_high",
            "up_from_30d_low",    "up_from_90d_low",
            "up_from_180d_low",   "up_from_365d_low",
        ],

        # Expanding historical extremes (lifetime context, always causal)
        "drawdown_expanding": [
            "ath_to_date", "atl_to_date",
            "dist_from_ath", "dist_from_atl",
        ],

        # Market structure (swing counts, trend persistence — all causal)
        "structure": [
            "lower_low_count_20d", "higher_low_count_20d",
            "lower_high_count_20d", "higher_high_count_20d",
            "trend_persistence_20d",
        ],

        # Extended momentum
        "momentum_extended": [
            "mom_30d", "mom_60d", "mom_90d",
            "rsi_change_7d", "macd_accel_3d",
            "adx14", "adx_change_7d",
        ],

        # Extended volume
        "volume_extended": [
            "vol_zscore_30d", "taker_buy_ratio",
        ],

        # Extended volatility
        "volatility_extended": [
            "vol_change_14d", "bb_percentile_90d",
        ],

        # Extended relative strength
        "macro_extended": [
            "rel_strength_eth_14d",
        ],
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

        # --- 5. Drawdown & Liquidity Proxy (existing) ---
        rolling_max_30 = high.rolling(window=30, min_periods=10).max()
        rolling_max_90 = high.rolling(window=90, min_periods=20).max()
        feat["dd_from_30d_high"] = (close - rolling_max_30) / (rolling_max_30 + 1e-9)
        feat["dd_from_90d_high"] = (close - rolling_max_90) / (rolling_max_90 + 1e-9)
        dollar_vol = close * vol
        feat["amihud_illiquidity_proxy"] = (log_ret.abs() / (dollar_vol + 1e-5)).rolling(14).mean()

        # --- 5b. Extended Rolling Distance-to-Extreme (BR-002) ---
        # All use High for rolling max, Low for rolling min — strictly causal
        rolling_max_180 = high.rolling(window=180, min_periods=30).max()
        rolling_max_365 = high.rolling(window=365, min_periods=60).max()
        rolling_min_30  = low.rolling(window=30,  min_periods=10).min()
        rolling_min_90  = low.rolling(window=90,  min_periods=20).min()
        rolling_min_180 = low.rolling(window=180, min_periods=30).min()
        rolling_min_365 = low.rolling(window=365, min_periods=60).min()

        feat["dd_from_180d_high"] = (close - rolling_max_180) / (rolling_max_180 + 1e-9)
        feat["dd_from_365d_high"] = (close - rolling_max_365) / (rolling_max_365 + 1e-9)
        feat["up_from_30d_low"]   = (close - rolling_min_30)  / (rolling_min_30  + 1e-9)
        feat["up_from_90d_low"]   = (close - rolling_min_90)  / (rolling_min_90  + 1e-9)
        feat["up_from_180d_low"]  = (close - rolling_min_180) / (rolling_min_180 + 1e-9)
        feat["up_from_365d_low"]  = (close - rolling_min_365) / (rolling_min_365 + 1e-9)

        # --- 5c. Expanding ATH/ATL — CAUSAL (BR-002) ---
        # expanding().max/min uses only data at or before t; never full-dataset values.
        ath = high.expanding().max()   # causal: max(High_{1..t})
        atl = low.expanding().min()    # causal: min(Low_{1..t})
        feat["ath_to_date"]   = ath
        feat["atl_to_date"]   = atl
        feat["dist_from_ath"] = (close - ath) / (ath + 1e-9)  # always <= 0
        feat["dist_from_atl"] = (close - atl) / (atl + 1e-9)  # always >= 0

        # --- 5d. Market Structure: swing counts (causal, 20-bar window) ---
        # Count directional swings in the rolling window — no future data
        local_high_20 = high.rolling(3, center=False).max()  # causal local high
        local_low_20  = low.rolling(3, center=False).min()

        def _swing_counts(series: pd.Series, window: int = 20) -> tuple:
            """Count lower-lows, higher-lows, lower-highs, higher-highs in rolling window."""
            n = len(series)
            ll = hh = lh = hl = trend = np.zeros(n)
            for i in range(window, n):
                w = series.iloc[i - window:i].values
                diffs = np.diff(w)
                ll[i] = float(np.sum(diffs < 0))
                hh[i] = float(np.sum(diffs > 0))
                lh[i] = float(np.sum(diffs < 0))   # for highs
                hl[i] = float(np.sum(diffs > 0))   # for lows
                trend[i] = float(np.mean(np.sign(diffs)))
            return ll, hh, lh, hl, trend

        ll_c, hh_c, _, _, _ = _swing_counts(close, window=20)
        _, _, lh_h, hl_h, trend_p = _swing_counts(local_high_20, window=20)

        feat["lower_low_count_20d"]   = pd.Series(ll_c, index=df.index)
        feat["higher_high_count_20d"] = pd.Series(hh_c, index=df.index)
        feat["lower_high_count_20d"]  = pd.Series(lh_h, index=df.index)
        feat["higher_low_count_20d"]  = pd.Series(hl_h, index=df.index)
        feat["trend_persistence_20d"] = pd.Series(trend_p, index=df.index)

        # --- 5e. Extended Momentum (BR-002) ---
        feat["mom_30d"] = (close - close.shift(30)) / (close.shift(30) + 1e-9)
        feat["mom_60d"] = (close - close.shift(60)) / (close.shift(60) + 1e-9)
        feat["mom_90d"] = (close - close.shift(90)) / (close.shift(90) + 1e-9)

        rsi14 = cls._compute_rsi(close, 14)
        feat["rsi_change_7d"] = rsi14 - rsi14.shift(7)

        # MACD acceleration (second derivative of histogram)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_raw = macd_line - signal_line
        feat["macd_accel_3d"] = (
            (macd_hist_raw - macd_hist_raw.shift(3)) -
            (macd_hist_raw.shift(3) - macd_hist_raw.shift(6))
        ) / (close + 1e-9)

        # ADX (Average Directional Index) — causal, period=14
        atr14 = cls._compute_atr(high, low, close, 14)
        plus_dm  = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
        smooth_plus  = plus_dm.ewm(alpha=1/14,  min_periods=14, adjust=False).mean()
        smooth_minus = minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        di_plus  = 100 * smooth_plus  / (atr14 + 1e-9)
        di_minus = 100 * smooth_minus / (atr14 + 1e-9)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
        adx14 = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        feat["adx14"]        = adx14
        feat["adx_change_7d"] = adx14 - adx14.shift(7)

        # --- 5f. Extended Volume (BR-002) ---
        vol_mean_30 = vol.rolling(30, min_periods=10).mean()
        vol_std_30  = vol.rolling(30, min_periods=10).std()
        feat["vol_zscore_30d"] = (vol - vol_mean_30) / (vol_std_30 + 1e-9)

        # Taker-buy ratio (proportion of volume initiated by buyers)
        if "taker_buy_base_asset_volume" in df.columns:
            taker_buy = df["taker_buy_base_asset_volume"].astype(float)
            feat["taker_buy_ratio"] = taker_buy / (vol + 1e-9)
        else:
            feat["taker_buy_ratio"] = 0.5  # neutral fallback

        # --- 5g. Extended Volatility (BR-002) ---
        realized_vol_14 = log_ret.rolling(window=14, min_periods=7).std() * np.sqrt(365)
        feat["vol_change_14d"] = realized_vol_14 - realized_vol_14.shift(14)

        sma20   = close.rolling(window=20, min_periods=10).mean()
        std20   = close.rolling(window=20, min_periods=10).std()
        bb_w    = (4.0 * std20) / (sma20 + 1e-9)
        bb_hist = bb_w.rolling(window=90, min_periods=20)
        feat["bb_percentile_90d"] = bb_hist.apply(
            lambda x: float(np.mean(x < x.iloc[-1])) if len(x) > 1 else 0.5,
            raw=False
        )

        # --- 6. Macro & Relative Context ---
        if btc_klines and len(btc_klines) >= len(klines):
            btc_closes = pd.Series([float(k[4]) for k in btc_klines[-len(klines):]], index=df.index)
            btc_ret_7d  = (btc_closes - btc_closes.shift(7))  / (btc_closes.shift(7)  + 1e-9)
            btc_ret_14d = (btc_closes - btc_closes.shift(14)) / (btc_closes.shift(14) + 1e-9)
            feat["rel_strength_btc_7d"]  = feat["mom_7d"]  - btc_ret_7d
            feat["rel_strength_btc_14d"] = feat["mom_14d"] - btc_ret_14d
        else:
            feat["rel_strength_btc_7d"]  = feat["mom_7d"]  * 0.5
            feat["rel_strength_btc_14d"] = feat["mom_14d"] * 0.5

        if eth_klines and len(eth_klines) >= len(klines):
            eth_closes  = pd.Series([float(k[4]) for k in eth_klines[-len(klines):]], index=df.index)
            eth_ret_7d  = (eth_closes - eth_closes.shift(7))  / (eth_closes.shift(7)  + 1e-9)
            eth_ret_14d = (eth_closes - eth_closes.shift(14)) / (eth_closes.shift(14) + 1e-9)
            feat["rel_strength_eth_7d"]  = feat["mom_7d"]  - eth_ret_7d
            feat["rel_strength_eth_14d"] = feat["mom_14d"] - eth_ret_14d
        else:
            feat["rel_strength_eth_7d"]  = feat["mom_7d"]  * 0.4
            feat["rel_strength_eth_14d"] = feat["mom_14d"] * 0.4

        # --- 7. Breadth Proxy ---
        feat["market_breadth_proxy"] = float(breadth_ratio)

        # Clean fill — numeric only, preserve timestamp and close
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
