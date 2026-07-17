import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from scorer import _compute_rsi, _stage1_score, score_candidates


def _row(index: int, amount: int = 100_000_000_000, change: float = 1.0) -> dict:
    return {
        "티커": f"{index:06d}",
        "종목명": f"종목{index}",
        "현재가": 10_000,
        "등락률": change,
        "거래대금": amount,
        "시가총액": 1_000_000_000_000,
        "거래량": 1_000,
    }


class ScorerRegressionTests(unittest.TestCase):
    def test_flat_prices_have_neutral_rsi(self):
        closes = pd.Series([100.0] * 30)
        self.assertEqual(_compute_rsi(closes), 50.0)

    def test_large_daily_drop_is_penalized_and_explained(self):
        score, _, risks = _stage1_score(_row(1, change=-12.0))
        self.assertLess(score, 0)
        self.assertTrue(any("급락" in text for text, _ in risks))

    def test_stage1_ties_prefer_higher_liquidity(self):
        rows = [_row(i, amount=100_000_000_000 + i) for i in range(31)]
        rows[-1]["거래대금"] = 10_000_000_000_000

        result = score_candidates(pd.DataFrame(rows), bars_provider=lambda _: None)

        self.assertIn(rows[-1]["티커"], {item["티커"] for item in result})

    def test_stage2_failure_is_exposed_in_candidate_result(self):
        def broken_provider(_):
            raise RuntimeError("daily bars failed")

        result = score_candidates(pd.DataFrame([_row(1)]), bars_provider=broken_provider)

        self.assertEqual(result[0]["2단계상태"], "조회 실패")
        self.assertIn("RuntimeError", result[0]["2단계오류"])
        self.assertFalse(result[0]["2단계완료"])
        self.assertEqual(result[0]["2단계완료수"], 0)
        self.assertEqual(result[0]["2단계대상수"], 1)

    def test_final_ranking_uses_score_before_data_availability(self):
        high_score = _row(1, amount=200_000_000_000, change=3.0)
        low_score = _row(2, amount=5_000_000_000, change=0.0)
        dates = pd.date_range("2026-05-01", periods=30, freq="B")
        flat_bars = pd.DataFrame({"Close": [100.0] * 30, "Volume": [1_000] * 30}, index=dates)

        result = score_candidates(
            pd.DataFrame([high_score, low_score]),
            bars_provider=lambda ticker: flat_bars if ticker == low_score["티커"] else None,
        )

        self.assertEqual(result[0]["티커"], high_score["티커"])

    def test_partial_daily_bar_is_excluded_from_all_technical_indicators(self):
        dates = pd.date_range("2026-06-08", periods=30, freq="B")
        bars = pd.DataFrame({
            "Close": [100.0] * 29 + [200.0],
            "Volume": [1_000] * 29 + [100_000],
        }, index=dates)
        fixed_now = datetime(2026, 7, 17, 10, 0)

        with patch("scorer.datetime") as mocked_datetime:
            mocked_datetime.now.return_value = fixed_now
            result = score_candidates(pd.DataFrame([_row(1)]), bars_provider=lambda _: bars)

        self.assertTrue(result[0]["미완성봉가능"])
        self.assertEqual(result[0]["수집일봉기준일"], "2026-07-17")
        self.assertEqual(result[0]["일봉기준일"], "2026-07-16")
        self.assertFalse(any("거래량 급증" in text for text, _ in result[0]["후보이유"]))

    def test_stage2_result_carries_daily_bar_date(self):
        dates = pd.date_range("2026-06-01", periods=30, freq="B")
        bars = pd.DataFrame({
            "Close": [100 + i for i in range(30)],
            "Volume": [1_000] * 30,
        }, index=dates)

        result = score_candidates(
            pd.DataFrame([_row(1)]),
            bars_provider=lambda _: bars,
            snapshot_date="2026-07-17",
        )

        self.assertTrue(result[0]["2단계완료"])
        self.assertEqual(result[0]["일봉기준일"], dates[-1].strftime("%Y-%m-%d"))
        self.assertTrue(result[0]["기준일불일치"])
        self.assertEqual(result[0]["관찰유형"], "추세 지속 관찰")


if __name__ == "__main__":
    unittest.main()
