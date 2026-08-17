"""
Volume Spread Analysis (VSA) Engine - Production Grade.
Ported and enhanced from VSA-2.0 and grounded in Tom Williams' 'Master the Markets'.
Implements:
1. Spread Z-Score Normalization (spread_z)
2. Close Location Value (CLV) & Candlestick Wick Rejection
3. Pink Volume Filter with Rolling Liquidity Floor
4. 8 Core High-Conviction VSA Patterns (Signs of Strength & Signs of Weakness)
5. Initiative Absorption (Effort and Result in Harmony)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)


@dataclass
class VSAParams:
    """Parameters for Volume Spread Analysis detection."""
    lookback: int = 20                   # Rolling window for baseline stats
    vol_extreme: float = 2.0             # Ultra-high volume threshold
    vol_high: float = 1.35               # High volume threshold
    vol_low: float = 0.70                # Low volume threshold
    spread_wide_z: float = 0.75          # Z-score for wide spread
    spread_ultra_wide_z: float = 1.50    # Z-score for ultra-wide spread
    spread_narrow_z: float = -0.50       # Z-score for narrow spread
    wick_rejection_ratio: float = 0.35   # Minimum wick ratio for rejection tails
    min_liquidity_quantile: float = 0.10 # Rolling quantile floor for low-volume tests
    min_liquidity_vol_rel: float = 0.30  # Absolute minimum relative volume floor


class VSAEngine:
    """
    Computes continuous VSA variables and detects discrete Signs of Strength (SoS)
    and Signs of Weakness (SoW) with mathematical rigor and zero lookahead bias.
    """

    def __init__(self, params: VSAParams = VSAParams()):
        self.params = params

    def compute_vsa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes full VSA feature suite and pattern flags on input OHLCV dataframe.
        """
        data = df.copy()
        high = data["high"]
        low = data["low"]
        open_p = data["open"]
        close = data["close"]
        vol = data["tick_volume"] if "tick_volume" in data.columns else data["volume"]

        # 1. Price Spread Calculation & Z-Score Normalization
        bar_range = np.maximum(high - low, 1e-4)
        spread_sma = bar_range.rolling(window=self.params.lookback, min_periods=5).mean()
        spread_std = bar_range.rolling(window=self.params.lookback, min_periods=5).std().fillna(0.10)
        spread_std_adj = np.maximum(spread_std, 0.10 * np.maximum(spread_sma, 0.10))
        spread_z = (bar_range - spread_sma) / spread_std_adj

        # 2. Close Location Value (CLV)
        clv = (close - low) / bar_range

        # 3. Candlestick Wick Ratios
        lower_wick = np.minimum(open_p, close) - low
        upper_wick = high - np.maximum(open_p, close)
        lower_wick_ratio = lower_wick / bar_range
        upper_wick_ratio = upper_wick / bar_range

        # 4. Volume Analysis & Pink Volume Filter
        vol_sma = vol.rolling(window=self.params.lookback, min_periods=5).mean()
        vol_rel = vol / np.maximum(vol_sma, 1e-4)

        prev_vol = vol.shift(1)
        prev_vol2 = vol.shift(2)
        raw_vol_pink = (vol < prev_vol) & (vol < prev_vol2)
        vol_quantile_floor = vol.rolling(window=self.params.lookback, min_periods=5).quantile(self.params.min_liquidity_quantile)
        min_liq = (vol_rel >= self.params.min_liquidity_vol_rel) | (vol >= vol_quantile_floor)
        vol_pink = raw_vol_pink & min_liq

        is_down_bar = close < close.shift(1)
        is_up_bar = close > close.shift(1)

        # =========================================================================
        # 5. SIGNS OF STRENGTH (SOS)
        # =========================================================================

        # SOS-1: No Supply Test / Low-Vol Test (Down bar on low drying volume)
        sos_no_supply = is_down_bar & ((vol_rel <= self.params.vol_low) | vol_pink) & (clv >= 0.45)

        # SOS-2: Stopping Volume / Absorption (High volume down-bar absorbed into support)
        sos_stopping_volume = is_down_bar & (vol_rel >= self.params.vol_high) & (clv >= 0.40) & (spread_z <= 1.20)

        # SOS-3: Shakeout / Reverse Upthrust (Lower wick rejection)
        sos_shakeout = (lower_wick_ratio >= self.params.wick_rejection_ratio) & (clv >= 0.60) & (vol_rel >= 1.10)

        # SOS-4: Selling Climax (Ultra-wide spread down on extreme volume closing off lows)
        sos_selling_climax = is_down_bar & (spread_z >= self.params.spread_ultra_wide_z) & (vol_rel >= self.params.vol_extreme) & (clv >= 0.35)

        # =========================================================================
        # 6. SIGNS OF WEAKNESS (SOW)
        # =========================================================================

        # SOW-1: No Demand Bar (Up bar on low drying volume)
        sow_no_demand = is_up_bar & ((vol_rel <= self.params.vol_low) | vol_pink) & (clv <= 0.55)

        # SOW-2: Upthrust (Upper wick rejection on high volume or trap volume)
        sow_upthrust = (upper_wick_ratio >= self.params.wick_rejection_ratio) & (clv <= 0.35)

        # SOW-3: Buying Climax (Ultra-wide spread up on extreme volume closing off highs)
        sow_buying_climax = is_up_bar & (spread_z >= self.params.spread_ultra_wide_z) & (vol_rel >= self.params.vol_extreme) & (clv <= 0.65)

        # SOW-4: End of a Rising Market (Narrow spread on high volume: hidden selling)
        sow_end_of_rising_market = is_up_bar & (spread_z <= self.params.spread_narrow_z) & (vol_rel >= self.params.vol_extreme) & (clv <= 0.60)

        # =========================================================================
        # 7. INITIATIVE ABSORPTION BREAKOUT (EFFORT & RESULT IN HARMONY)
        # =========================================================================
        bullish_absorption = is_up_bar & (spread_z >= 0.50) & (vol_rel >= self.params.vol_high) & (clv >= 0.65)
        bearish_absorption = is_down_bar & (spread_z >= 0.50) & (vol_rel >= self.params.vol_high) & (clv <= 0.35)

        # Output Dataframe
        vsa_df = pd.DataFrame(index=data.index)
        vsa_df["vsa_spread_z"] = spread_z
        vsa_df["vsa_clv"] = clv
        vsa_df["vsa_vol_rel"] = vol_rel
        vsa_df["vsa_vol_pink"] = vol_pink
        vsa_df["vsa_lower_wick_ratio"] = lower_wick_ratio
        vsa_df["vsa_upper_wick_ratio"] = upper_wick_ratio

        # SOS Flags
        vsa_df["vsa_sos_no_supply"] = sos_no_supply
        vsa_df["vsa_sos_stopping_volume"] = sos_stopping_volume
        vsa_df["vsa_sos_shakeout"] = sos_shakeout
        vsa_df["vsa_sos_selling_climax"] = sos_selling_climax
        vsa_df["vsa_any_sos"] = (
            sos_no_supply | sos_stopping_volume | sos_shakeout | sos_selling_climax
        )

        # SOW Flags
        vsa_df["vsa_sow_no_demand"] = sow_no_demand
        vsa_df["vsa_sow_upthrust"] = sow_upthrust
        vsa_df["vsa_sow_buying_climax"] = sow_buying_climax
        vsa_df["vsa_sow_end_of_rising_market"] = sow_end_of_rising_market
        vsa_df["vsa_any_sow"] = (
            sow_no_demand | sow_upthrust | sow_buying_climax | sow_end_of_rising_market
        )

        vsa_df["vsa_bullish_absorption"] = bullish_absorption
        vsa_df["vsa_bearish_absorption"] = bearish_absorption

        return vsa_df
