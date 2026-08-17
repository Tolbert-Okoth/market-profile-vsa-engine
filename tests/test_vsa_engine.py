"""
Unit Tests for Refined Production-Grade Volume Spread Analysis (VSA) Engine.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.models.vsa_engine import VSAEngine, VSAParams


def generate_mock_vsa_df(n: int = 30) -> pd.DataFrame:
    """Generates synthetic baseline M5 bars for testing VSA detection."""
    base_time = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    times = [base_time + timedelta(minutes=5 * i) for i in range(n)]
    
    opens = np.full(n, 2000.0)
    highs = np.full(n, 2000.5)
    lows = np.full(n, 1999.5)
    closes = np.full(n, 2000.2)
    volumes = np.full(n, 1000.0)
    spreads = np.full(n, 15.0)

    df = pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes,
        "spread": spreads
    })
    return df


def test_vsa_no_supply_test_detection():
    """Validates detection of No Supply Test (dip on pink volume closing in upper half)."""
    df = generate_mock_vsa_df(25)
    
    df.loc[24, "open"] = 2000.0
    df.loc[24, "high"] = 2000.2
    df.loc[24, "low"] = 1999.2
    df.loc[24, "close"] = 1999.8  # Down-bar closing in upper half (clv = 0.60)
    df.loc[24, "tick_volume"] = 400.0  # Pink volume (< prev 2)

    vsa = VSAEngine()
    res = vsa.compute_vsa_features(df)
    
    assert res.loc[24, "vsa_sos_no_supply"] == True
    assert res.loc[24, "vsa_any_sos"] == True


def test_vsa_upthrust_detection():
    """Validates detection of Upthrust (probe above high on wide upper wick closing near lows)."""
    df = generate_mock_vsa_df(25)
    
    df.loc[24, "open"] = 2000.0
    df.loc[24, "high"] = 2003.0  # Upper probe
    df.loc[24, "low"] = 1999.8
    df.loc[24, "close"] = 2000.1  # Closes near bottom (clv <= 0.35)
    df.loc[24, "tick_volume"] = 1800.0

    vsa = VSAEngine()
    res = vsa.compute_vsa_features(df)
    
    assert res.loc[24, "vsa_sow_upthrust"] == True
    assert res.loc[24, "vsa_any_sow"] == True


def test_vsa_stopping_volume_detection():
    """Validates detection of Stopping Volume (high volume down-bar absorbed into support)."""
    df = generate_mock_vsa_df(25)
    
    df.loc[24, "open"] = 2000.0
    df.loc[24, "high"] = 2000.1
    df.loc[24, "low"] = 1999.0
    df.loc[24, "close"] = 1999.6  # Down-bar closing off lows (clv = 0.54)
    df.loc[24, "tick_volume"] = 2000.0  # High volume absorption

    vsa = VSAEngine()
    res = vsa.compute_vsa_features(df)
    
    assert res.loc[24, "vsa_sos_stopping_volume"] == True
    assert res.loc[24, "vsa_any_sos"] == True
