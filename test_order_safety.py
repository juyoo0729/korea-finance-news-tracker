import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import bithumb_client
import toss_client
from order_safety import record_order_attempt, was_order_attempted


class OrderSafetyTests(unittest.TestCase):
    def test_dashboard_requires_explicit_toss_account_selection(self):
        source = Path("dashboard.py").read_text(encoding="utf-8")
        start = source.index('st.selectbox(\n                        "주문 계좌"')
        account_widget = source[start : start + 300]
        self.assertIn("index=None", account_widget)

    def test_toss_order_requires_explicit_account(self):
        with patch.object(toss_client, "_access_token", return_value="token"), patch.object(
            toss_client.requests, "post"
        ) as post:
            result = toss_client.place_stock_order("005930", "BUY", 1, 70000)

        self.assertFalse(result["ok"])
        self.assertIn("계좌", result["data"]["error"])
        post.assert_not_called()

    def test_toss_order_sends_account_and_idempotency_key(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"result": {"orderId": "1"}}
        with patch.object(toss_client, "_access_token", return_value="token"), patch.object(
            toss_client.requests, "post", return_value=response
        ) as post:
            result = toss_client.place_stock_order(
                "005930",
                "BUY",
                1,
                70000,
                account_seq=123,
                client_order_id="abc123",
            )

        self.assertTrue(result["ok"])
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["X-Tossinvest-Account"], "123")
        self.assertEqual(kwargs["json"]["clientOrderId"], "abc123")

    def test_bithumb_order_sends_identifier(self):
        response = Mock(ok=True, status_code=201)
        response.json.return_value = {"uuid": "order-1"}
        with patch.object(bithumb_client, "_jwt", return_value="token") as jwt, patch.object(
            bithumb_client.requests, "post", return_value=response
        ) as post:
            result = bithumb_client.place_order(
                "KRW-BTC",
                "bid",
                "limit",
                volume="0.001",
                price="100000000",
                identifier="abc123",
            )

        self.assertTrue(result["ok"])
        params = post.call_args.kwargs["json"]
        self.assertEqual(params["identifier"], "abc123")
        jwt.assert_called_once_with(params)

    def test_bithumb_timeout_returns_unknown_result(self):
        with patch.object(bithumb_client, "_jwt", return_value="token"), patch.object(
            bithumb_client.requests, "post", side_effect=requests.Timeout("timed out")
        ):
            result = bithumb_client.place_order(
                "KRW-BTC", "bid", "limit", "0.001", "100000000", identifier="abc"
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 0)

    def test_attempt_registry_blocks_a_b_a_sequence(self):
        state = {}
        record_order_attempt(state, "A")
        record_order_attempt(state, "B")

        self.assertTrue(was_order_attempted(state, "A"))
        self.assertTrue(was_order_attempted(state, "B"))


if __name__ == "__main__":
    unittest.main()
