import unittest
from unittest.mock import patch

import pandas as pd

import market_data


class MarketDataRegressionTests(unittest.TestCase):
    def test_latest_trading_date_comes_from_market_index(self):
        index = pd.to_datetime(["2026-07-15", "2026-07-16"])
        frame = pd.DataFrame({"Close": [3000, 3010]}, index=index)

        with patch.object(market_data.fdr, "DataReader", return_value=frame):
            actual = market_data.get_latest_trading_date()

        self.assertEqual(actual, "2026-07-16")

    def test_latest_trading_date_is_unknown_when_index_lookup_is_empty(self):
        with patch.object(market_data.fdr, "DataReader", return_value=pd.DataFrame()):
            actual = market_data.get_latest_trading_date()

        self.assertIsNone(actual)

    def test_snapshot_key_changes_when_market_values_change(self):
        first = pd.DataFrame([
            {"티커": "000001", "현재가": 1000, "등락률": 1.0, "거래대금": 100},
        ])
        second = first.copy()
        second.loc[0, "현재가"] = 1010

        self.assertNotEqual(
            market_data.market_snapshot_key(first, "2026-07-16"),
            market_data.market_snapshot_key(second, "2026-07-16"),
        )


if __name__ == "__main__":
    unittest.main()
