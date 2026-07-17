import unittest
from unittest.mock import Mock, patch

import bithumb_client
import toss_client


class OrderSafetyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
