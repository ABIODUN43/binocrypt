"""
Unit tests for Universe providers (SBRU_V1 and PIT_V2 interface).
"""

from datetime import date
import pytest
from app.research.universe.providers import (
    SBRUProvider,
    SBRU_V1_SYMBOLS,
    SBRU_V1_LISTED_FROM,
    PITUniverseProvider,
)

def test_sbru_provider():
    sbru = SBRUProvider()
    assert sbru.universe_type == "SBRU_V1"
    assert sbru.survivorship_bias_acknowledged is True
    
    # Pre-2022 -> empty set
    assert len(sbru.eligible_symbols(date(2021, 12, 31))) == 0
    # Post-2022 -> full set
    eligible = sbru.eligible_symbols(date(2022, 1, 1))
    assert len(eligible) == len(SBRU_V1_SYMBOLS)
    assert "ARBUSDT" in eligible
    assert "BTCUSDT" in eligible

def test_pit_provider_stub():
    pit = PITUniverseProvider()
    assert pit.universe_type == "PIT_V2"
    with pytest.raises(RuntimeError):
        pit.eligible_symbols(date(2023, 1, 1))
