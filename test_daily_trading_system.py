import unittest

import numpy as np
import pandas as pd

from daily_trading_system import (
    add_indicators,
    classify_watchlist_position,
    compute_canslim_proxy_score,
    score_market_regime,
)


def make_ohlcv(closes, volumes=None):
    closes = pd.Series(closes, index=pd.date_range("2024-01-01", periods=len(closes), freq="B"))
    if volumes is None:
        volumes = pd.Series(np.full(len(closes), 2_000_000), index=closes.index)
    else:
        volumes = pd.Series(volumes, index=closes.index)
    frame = pd.DataFrame(
        {
            "Open": closes * 0.995,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": volumes,
        }
    )
    return frame


class DailyTradingSystemTests(unittest.TestCase):
    def test_score_market_regime_flags_bullish(self):
        spy = make_ohlcv(np.linspace(400, 500, 260))
        qqq = make_ohlcv(np.linspace(300, 430, 260))

        regime = score_market_regime({"SPY": spy, "QQQ": qqq})

        self.assertEqual(regime["state"], "BULLISH")
        self.assertTrue(regime["adds_allowed"])

    def test_compute_canslim_proxy_score_rewards_growth_and_leadership(self):
        closes = list(np.linspace(50, 118, 259)) + [126.0]
        volumes = [3_000_000] * 259 + [8_500_000]
        stock = add_indicators(make_ohlcv(closes, volumes=volumes))
        benchmark = make_ohlcv(np.linspace(400, 450, 260))
        info = {
            "earningsQuarterlyGrowth": 0.42,
            "revenueGrowth": 0.31,
            "returnOnEquity": 0.22,
            "heldPercentInstitutions": 0.58,
            "floatShares": 250_000_000,
            "marketCap": 18_000_000_000,
        }

        result = compute_canslim_proxy_score(
            ticker="TEST",
            frame=stock,
            benchmark=benchmark,
            info=info,
            market_regime={"state": "BULLISH"},
            ep_signal={"is_ep": True, "ep_score": 60, "gap_percent": 12.5},
        )

        self.assertGreaterEqual(result["canslim_score"], 70)
        self.assertTrue(result["breakout_ready"])

    def test_classify_watchlist_position_flags_breakout_add(self):
        closes = list(np.linspace(50, 95, 239)) + [95.2 + (i * 0.05) for i in range(20)] + [101.0]
        volumes = [2_000_000] * 259 + [7_000_000]
        frame = make_ohlcv(closes, volumes=volumes)

        result = classify_watchlist_position(
            ticker="TEST",
            frame=frame,
            market_regime={"state": "BULLISH", "adds_allowed": True},
            cost_basis=75.0,
            shares=100,
        )

        self.assertEqual(result["action"], "ADD_ON_BREAKOUT")

    def test_classify_watchlist_position_flags_exit(self):
        uptrend = list(np.linspace(50, 120, 240))
        breakdown = [118, 116, 114, 111, 108, 104, 98, 92, 88, 82, 76, 71, 66, 62, 58, 54, 51, 48, 46, 44]
        closes = uptrend + breakdown
        volumes = [2_500_000] * 240 + [4_000_000] * 20
        frame = make_ohlcv(closes, volumes=volumes)

        result = classify_watchlist_position(
            ticker="TEST",
            frame=frame,
            market_regime={"state": "CAUTION", "adds_allowed": False},
            cost_basis=80.0,
            shares=100,
        )

        self.assertEqual(result["action"], "EXIT")


if __name__ == "__main__":
    unittest.main()
