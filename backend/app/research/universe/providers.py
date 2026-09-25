"""
Universe providers for BR-002 / BR-003.

Design
------
All downstream code (feature builder, label generator, model trainer) calls
    universe.eligible_symbols(as_of_date)
Swapping SBRU_V1 for a true PIT_V2 universe requires no other changes.

Phase 1 — SBRU_V1 (survivorship-biased reference universe)
    Static list of consistently-listed USDT pairs. Symbols absent from
    today's Binance spot market are excluded. Survivorship bias is present
    and must be disclosed in every research report.

Phase 2 — PITUniverseProvider (stub, to be completed)
    Will consume a listing/delisting CSV seeded from Binance announcements.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base — all universe providers implement this interface
# ---------------------------------------------------------------------------

class UniverseProvider(ABC):
    """
    Returns the set of symbols eligible at a given historical date.

    Implementors must guarantee that eligible_symbols(t) contains only
    symbols that were actually tradeable at time t — i.e., no look-ahead
    into future listings or delistings.
    """

    universe_type: str = "UNKNOWN"

    @abstractmethod
    def eligible_symbols(self, as_of_date: date) -> Set[str]:
        """Returns the set of tradeable symbols as of `as_of_date`."""
        ...

    @abstractmethod
    def all_symbols(self) -> Set[str]:
        """Returns all symbols ever covered by this universe."""
        ...


# ---------------------------------------------------------------------------
# SBRU_V1 — Phase 1 Survivorship-Biased Reference Universe
# ---------------------------------------------------------------------------

# ~100 consistently-listed large and mid-cap USDT pairs on Binance.
# Criteria: present on Binance spot market through the full 2022–2025 window.
# Symbols that were listed after 2022-01-01 or delisted before 2025-01-01
# are excluded, which INTRODUCES SURVIVORSHIP BIAS.
# This is acknowledged and must appear in all research reports.

SBRU_V1_SYMBOLS: Set[str] = {
    # Majors
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    # Large-cap L1
    "SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT",
    "LTCUSDT", "LINKUSDT", "UNIUSDT", "XLMUSDT", "VETUSDT", "FILUSDT",
    "TRXUSDT", "ICPUSDT", "HBARUSDT", "ALGOUSDT", "XTZUSDT", "EGLDUSDT",
    # Mid-cap L1/L2
    "ARBUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "FTMUSDT", "FLOWUSDT",
    "KLAYUSDT", "RUNEUSDT", "KASUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
    "TIAUSDT", "MINAUSDT", "IMXUSDT", "GALAUSDT", "SANDUSDT", "MANAUSDT",
    "AXSUSDT", "APEUSDT", "GMTUSDT",
    # DeFi
    "AAVEUSDT", "MKRUSDT", "COMPUSDT", "SNXUSDT", "CRVUSDT", "BALUSDT",
    "YFIUSDT", "SUSHIUSDT", "1INCHUSDT", "DYDXUSDT", "GMXUSDT",
    # Infrastructure / Interop
    "QNTUSDT", "ZECUSDT", "DASHUSDT", "XMRUSDT", "IOTAUSDT",
    "ONTUSDT", "ZILUSDT", "BATUSDT", "STORJUSDT", "ANKRUSDT",
    # Gaming / NFT
    "ENJUSDT", "CHZUSDT", "RNDRUSDT", "LRCUSDT",
    # Oracle / Data
    "BANDUSDT", "APIUSDT",
    # Exchange tokens
    "CAKEUSDT", "GRTUSDT",
    # Meme / Community
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT",
    # Other consistently-listed
    "ETCUSDT", "BCHUSDT", "XRPUSDT", "EOSUSDT", "IOSTUSDT",
    "NEOUSDT", "WAVESUSDT", "ZRXUSDT", "KNCUSDT", "COTIUSDT",
    "RVNUSDT", "CELRUSDT", "FETUSDT", "OCEANUSDT", "NUUSDT",
    "ARUSDT", "FLMUSDT",
}

# Approximate date from which all SBRU_V1 symbols were consistently listed.
# This is an approximation — the true listing dates vary.
SBRU_V1_LISTED_FROM = date(2022, 1, 1)


class SBRUProvider(UniverseProvider):
    """
    Phase 1: Survivorship-Biased Reference Universe (SBRU_V1).

    IMPORTANT — Survivorship Bias Disclosure
    -----------------------------------------
    This universe contains only symbols consistently listed across the full
    study window. Assets that were listed and subsequently delisted are
    EXCLUDED. This may cause the research to overstate signal quality
    relative to a true point-in-time universe.

    All research reports must include:
        universe_type = "SBRU_V1"
        survivorship_bias_acknowledged = True

    Phase 2 will introduce PITUniverseProvider with historical listing/
    delisting records to quantify the magnitude of this bias.
    """

    universe_type: str = "SBRU_V1"
    survivorship_bias_acknowledged: bool = True

    def eligible_symbols(self, as_of_date: date) -> Set[str]:
        """
        Returns SBRU_V1 symbols eligible at `as_of_date`.

        Before SBRU_V1_LISTED_FROM: returns empty set (no history).
        On or after that date: returns the full static universe.

        NOTE: This does NOT correctly model entry/exit of individual symbols.
        It is a crude approximation intended only for Phase 1.
        """
        if as_of_date < SBRU_V1_LISTED_FROM:
            return set()
        return frozenset(SBRU_V1_SYMBOLS)

    def all_symbols(self) -> Set[str]:
        return frozenset(SBRU_V1_SYMBOLS)

    @property
    def bias_disclosure(self) -> str:
        return (
            "Universe SBRU_V1: survivorship-biased reference universe. "
            "Approximately 100 consistently-listed USDT pairs on Binance. "
            "Assets delisted before the end of the study window are EXCLUDED. "
            "Results may overstate signal quality vs. a true PIT universe. "
            "Phase 2 (PIT_V2) will quantify this bias."
        )


# ---------------------------------------------------------------------------
# PITUniverseProvider — Phase 2 stub
# ---------------------------------------------------------------------------

class PITUniverseProvider(UniverseProvider):
    """
    Phase 2: True Point-in-Time Universe.

    Reads a listing/delisting CSV with columns:
        symbol, listed_date, delisted_date (NaT if still active)

    Each call to eligible_symbols(t) returns only symbols that were actually
    tradeable on Binance at time t, using the recorded listing/delisting dates.

    This eliminates survivorship bias but requires a seeded listing history.
    """

    universe_type: str = "PIT_V2"

    def __init__(self, listing_csv: Optional[Path] = None):
        self._records: Optional[pd.DataFrame] = None
        if listing_csv is not None:
            self._load(listing_csv)

    def _load(self, path: Path) -> None:
        df = pd.read_csv(path, parse_dates=["listed_date", "delisted_date"])
        self._records = df

    def eligible_symbols(self, as_of_date: date) -> Set[str]:
        if self._records is None:
            raise RuntimeError(
                "PITUniverseProvider: listing CSV not loaded. "
                "Provide a listing_csv path to __init__."
            )
        ts = pd.Timestamp(as_of_date)
        mask = (
            (self._records["listed_date"] <= ts) &
            (self._records["delisted_date"].isna() |
             (self._records["delisted_date"] > ts))
        )
        return frozenset(self._records.loc[mask, "symbol"].tolist())

    def all_symbols(self) -> Set[str]:
        if self._records is None:
            return frozenset()
        return frozenset(self._records["symbol"].tolist())
