"""toss_client.py — 토스증권 Open API 시세·자산·주문 연동

환경변수 TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 필요.
실시간 시세와 계좌 자산을 조회하고, 사용자가 확인한 주문을 API로 전송한다.
"""
import os
import time

import requests

_BASE = "https://openapi.tossinvest.com"
_token = {"value": None, "exp": 0.0}


def _access_token() -> str | None:
    now = time.time()
    if _token["value"] and now < _token["exp"] - 60:
        return _token["value"]
    cid = os.getenv("TOSS_CLIENT_ID")
    sec = os.getenv("TOSS_CLIENT_SECRET")
    if not cid or not sec:
        return None
    r = requests.post(
        f"{_BASE}/oauth2/token",
        data={"grant_type": "client_credentials", "client_id": cid, "client_secret": sec},
        timeout=5,
    )
    r.raise_for_status()
    d = r.json()
    _token["value"] = d["access_token"]
    _token["exp"] = now + d.get("expires_in", 86400)
    return _token["value"]


def get_realtime_price(symbol: str) -> float | None:
    """토스 실시간 현재가(원). 키 없거나 실패 시 None."""
    try:
        tok = _access_token()
        if not tok:
            return None
        r = requests.get(
            f"{_BASE}/api/v1/prices",
            params={"symbols": symbol},
            headers={"Authorization": f"Bearer {tok}"},
            timeout=5,
        )
        r.raise_for_status()
        res = r.json().get("result") or []
        if res and res[0].get("lastPrice"):
            return float(res[0]["lastPrice"])
    except Exception:
        return None
    return None


def _account_seq(tok: str) -> int | None:
    r = requests.get(
        f"{_BASE}/api/v1/accounts",
        headers={"Authorization": f"Bearer {tok}"},
        timeout=5,
    )
    r.raise_for_status()
    res = r.json().get("result") or []
    return res[0]["accountSeq"] if res else None


def get_stock_assets() -> dict | None:
    """토스 주식 자산 요약: 평가액·예수금·보유종목. 키 없거나 실패 시 None."""
    try:
        tok = _access_token()
        if not tok:
            return None
        seq = _account_seq(tok)
        if seq is None:
            return None
        H = {"Authorization": f"Bearer {tok}", "X-Tossinvest-Account": str(seq)}
        hold = (requests.get(f"{_BASE}/api/v1/holdings", headers=H, timeout=5)
                .json().get("result") or {})
        bp = (requests.get(f"{_BASE}/api/v1/buying-power?currency=KRW", headers=H, timeout=5)
              .json().get("result") or {})
        mv = ((hold.get("marketValue") or {}).get("amount") or {}).get("krw") or 0
        return {
            "holdings_value": float(mv),
            "cash": float(bp.get("cashBuyingPower") or 0),
            "items": hold.get("items") or [],
        }
    except Exception:
        return None


def place_stock_order(symbol: str, side: str, quantity: int,
                      price: float | None = None, order_type: str = "LIMIT") -> dict:
    """토스 주식 주문. side: BUY/SELL, order_type: LIMIT(가격필수)/MARKET.

    반환: {"ok": bool, "status": int, "data": dict}
    """
    try:
        tok = _access_token()
        if not tok:
            return {"ok": False, "status": 0, "data": {"error": "API 키 미설정"}}
        seq = _account_seq(tok)
        if seq is None:
            return {"ok": False, "status": 0, "data": {"error": "계좌 없음"}}
        H = {"Authorization": f"Bearer {tok}", "X-Tossinvest-Account": str(seq)}
        body = {"symbol": symbol, "side": side, "orderType": order_type,
                "quantity": str(int(quantity))}
        if order_type == "LIMIT":
            body["price"] = str(int(price))
        r = requests.post(f"{_BASE}/api/v1/orders", json=body, headers=H, timeout=5)
        try:
            data = r.json()
        except Exception:
            data = {"error": r.text[:200]}
        return {"ok": r.ok, "status": r.status_code, "data": data}
    except Exception as e:
        return {"ok": False, "status": 0, "data": {"error": str(e)}}


if __name__ == "__main__":
    # 자가 점검: 삼성전자 실시간 시세 + 자산
    p = get_realtime_price("005930")
    print("삼성전자 실시간:", p, "원" if p else "(키 미설정 또는 실패)")
    a = get_stock_assets()
    print("주식자산:", a if a else "(키 미설정 또는 실패)")
