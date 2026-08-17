"""
Dalton Auction Market Theory & Volume Spread Analysis (VSA) Strategy Engine.
Implements:
1. Dalton 80% Rule Rotations with VSA Strength/Weakness confirmation (Tom Williams).
2. Point of Control (POC) Target Room & Dynamic Friction Gating.
3. Initial Balance Breakout with VSA Initiative Absorption Volume.
4. Institutional Session Gating & Multi-Scale Trend Alignment.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.models.market_profile import MarketProfileEngine, MarketProfileParams
from src.models.vsa_engine import VSAEngine, VSAParams


@dataclass
class DaltonStrategyParams:
    """Parameters for Dalton Auction Market Theory + VSA Strategy."""
    # Dalton 80% Rule
    consecutive_closes_inside_va: int = 4   # M5 bars inside VA (20 minutes)
    min_va_width: float = 6.0               # Minimum VA width in USD
    max_va_width: float = 60.0              # Maximum VA width in USD
    min_rejection_wick_ratio: float = 0.35  # Tail/wick rejection ratio (wicks >= 35%)
    min_poc_room_atr_mult: float = 1.0      # Minimum room from entry to prior POC (x ATR)
    min_poc_dist_usd: float = 0.30          # Minimum absolute distance to POC
    probe_lookback: int = 12                # Lookback bars for probe outside VA

    # Initial Balance Breakout
    ib_min_range: float = 4.0               # Minimum IB range in USD
    ib_max_range: float = 30.0              # Maximum IB range in USD
    ib_breakout_buffer: float = 0.30        # Buffer above/below IBH/IBL

    # Macro Trend Alignment
    ema_fast_span: int = 72                 # 6-hour EMA
    ema_slow_span: int = 288                # 24-hour EMA

    # Execution Sessions
    london_open_utc: float = 8.5            # 08:30 UTC
    london_close_utc: float = 12.0          # 12:00 UTC
    ny_open_utc: float = 13.5               # 13:30 UTC
    ny_close_utc: float = 17.5              # 17:30 UTC
    max_spread: float = 40.0                # Max spread in points (0.40 USD)


class DaltonAuctionStrategy:
    """
    Unified Dalton Auction Market Theory & Volume Spread Analysis Signal Generator.
    Combines Market Profile Value Areas (VAH/VAL/POC), Initial Balance (IBH/IBL),
    and Tom Williams' VSA Strength/Weakness patterns.
    """

    def __init__(self,
                 mp_params: MarketProfileParams = MarketProfileParams(),
                 vsa_params: VSAParams = VSAParams(),
                 strat_params: DaltonStrategyParams = DaltonStrategyParams()):
        self.mp_engine = MarketProfileEngine(mp_params)
        self.vsa_engine = VSAEngine(vsa_params)
        self.params = strat_params

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Computes causal Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def generate_auction_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes historical OHLCV data, builds causal daily Market Profiles,
        computes VSA features, and generates causal trade signals.
        """
        data = df.copy().reset_index(drop=True)
        
        # 1. Compute Causal Market Profile Reference Levels (VAH, VAL, POC, IBH, IBL)
        if "prev_vah" not in data.columns:
            mp_df = self.mp_engine.compute_rolling_market_profiles(data)
            data = pd.concat([data, mp_df], axis=1)

        # 2. Compute Volume Spread Analysis (VSA) Metrics
        if "vsa_any_sos" not in data.columns:
            vsa_df = self.vsa_engine.compute_vsa_features(data)
            data = pd.concat([data, vsa_df], axis=1)

        # 3. Technical Anchors & ATR
        data["atr"] = self._compute_atr(data)
        data["ema_6h"] = data["close"].ewm(span=self.params.ema_fast_span, adjust=False).mean()
        data["ema_24h"] = data["close"].ewm(span=self.params.ema_slow_span, adjust=False).mean()

        macro_bull = data["close"] > data["ema_24h"]
        macro_bear = data["close"] < data["ema_24h"]

        # 4. Institutional Liquidity Window & Spread Filter
        h = data["time"].dt.hour + data["time"].dt.minute / 60.0
        in_session = (
            ((h >= self.params.london_open_utc) & (h <= self.params.london_close_utc)) |
            ((h >= self.params.ny_open_utc) & (h <= self.params.ny_close_utc))
        )
        spread_ok = (data["spread"] <= self.params.max_spread) if "spread" in data.columns else np.ones(len(data), dtype=bool)
        valid_context = in_session & spread_ok & ~data["prev_vah"].isna() & ~data["prev_val"].isna() & ~data["prev_poc"].isna()

        # 5. Dalton 80% Rule Acceptance into Prior Value Area
        inside_va = (data["close"] >= data["prev_val"]) & (data["close"] <= data["prev_vah"])
        va_consec_inside = inside_va.groupby((~inside_va).cumsum()).cumsum()
        va_accepted = (va_consec_inside >= self.params.consecutive_closes_inside_va)
        va_width = data["prev_vah"] - data["prev_val"]
        va_width_valid = (va_width >= self.params.min_va_width) & (va_width <= self.params.max_va_width)

        # VSA Signs on Probes outside Value Area
        had_probe_below_val = (data["low"].rolling(self.params.probe_lookback).min() < data["prev_val"])
        had_probe_above_vah = (data["high"].rolling(self.params.probe_lookback).max() > data["prev_vah"])
        recent_sos = data["vsa_any_sos"].rolling(8).max() > 0
        recent_sow = data["vsa_any_sow"].rolling(8).max() > 0

        # 6. Point of Control (POC) Target Room Validation
        dist_to_poc = data["prev_poc"] - data["close"]
        dist_from_poc = data["close"] - data["prev_poc"]

        long_poc_room = (dist_to_poc >= self.params.min_poc_room_atr_mult * data["atr"]) & (data["close"] < (data["prev_poc"] - self.params.min_poc_dist_usd))
        short_poc_room = (dist_from_poc >= self.params.min_poc_room_atr_mult * data["atr"]) & (data["close"] > (data["prev_poc"] + self.params.min_poc_dist_usd))

        # 7. Initial Balance (IB) Breakout Tracker with VSA Absorption Volume
        ib_active = data["is_ib_complete"] & (data["curr_ibr"] >= self.params.ib_min_range) & (data["curr_ibr"] <= self.params.ib_max_range)
        ib_breakout_high = ib_active & (data["close"] > data["curr_ibh"] + self.params.ib_breakout_buffer) & data["vsa_bullish_absorption"] & macro_bull
        ib_breakout_low = ib_active & (data["close"] < data["curr_ibl"] - self.params.ib_breakout_buffer) & data["vsa_bearish_absorption"] & macro_bear

        # 8. Generate Directional Signals
        trade_signals = np.zeros(len(data))

        # Strategy 1: Dalton 80% Rule Rotations with VSA Confirmation
        long_80_rule = valid_context & va_accepted & va_width_valid & had_probe_below_val & recent_sos & long_poc_room
        short_80_rule = valid_context & va_accepted & va_width_valid & had_probe_above_vah & recent_sow & short_poc_room

        trade_signals[long_80_rule] = 1.0
        trade_signals[short_80_rule] = -1.0

        # Strategy 2: Initial Balance Initiative Absorption Breakout Trades
        long_ib_break = valid_context & ib_breakout_high & (trade_signals == 0)
        short_ib_break = valid_context & ib_breakout_low & (trade_signals == 0)

        trade_signals[long_ib_break] = 1.0
        trade_signals[short_ib_break] = -1.0

        data["trade_signal"] = trade_signals
        return data
