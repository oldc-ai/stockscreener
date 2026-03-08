import unittest

import pandas as pd

from bull_run_system import debounce, find_best_run, future_stats


class BullRunSystemTests(unittest.TestCase):
    def test_find_best_run_uses_prior_low_and_future_high(self):
        index = pd.date_range("2020-01-01", periods=7, freq="D")
        close = pd.Series([10.0, 8.0, 9.0, 7.0, 11.0, 21.0, 19.0], index=index)

        run = find_best_run(close)

        self.assertIsNotNone(run)
        self.assertEqual(run["run_start_date"], index[3])
        self.assertEqual(run["run_end_date"], index[5])
        self.assertAlmostEqual(run["max_return_factor"], 3.0)
        self.assertEqual(run["run_days"], 3)

    def test_debounce_keeps_first_signal_in_each_cooldown_window(self):
        index = pd.date_range("2020-01-01", periods=8, freq="D")
        raw = pd.Series([False, True, True, False, True, False, True, False], index=index)

        debounced = debounce(raw, cooldown=2)

        expected = pd.Series([False, True, False, False, True, False, True, False], index=index)
        pd.testing.assert_series_equal(debounced, expected)

    def test_future_stats_reports_return_gain_and_drawdown(self):
        index = pd.date_range("2020-01-01", periods=5, freq="D")
        close = pd.Series([10.0, 12.0, 9.0, 15.0, 14.0], index=index)

        stats = future_stats(close, index[0], window=4)

        self.assertAlmostEqual(stats["return_4d"], 0.4)
        self.assertAlmostEqual(stats["max_gain_4d"], 0.5)
        self.assertAlmostEqual(stats["max_drawdown_4d"], -0.1)


if __name__ == "__main__":
    unittest.main()
