"""bithumb_client.py — 빗썸 시세·잔고·주문 연동

- 시세: 공개 API (인증 불필요)
- 잔고·주문: 개인 API (BITHUMB_ACCESS_KEY / BITHUMB_SECRET_KEY 환경변수 필요, JWT 서명)
"""
import base64
import hashlib
import hmac
import json
import os
import time
import uuid

import requests

_BASE = "https://api.bithumb.com"


def get_krw_tickers() -> list[dict]:
    """KRW 마켓 전 종목 티커. 실패 시 빈 리스트."""
    try:
        markets = requests.get(
            f"{_BASE}/v1/market/all", params={"isDetails": "false"}, timeout=5
        ).json()
        names = {
            m["market"]: m["korean_name"]
            for m in markets
            if m["market"].startswith("KRW-")
        }
        if not names:
            return []
        codes = list(names)
        data = []
        for i in range(0, len(codes), 100):  # URL 길이 제한 → 100개씩 배치
            chunk = codes[i:i + 100]
            data += requests.get(
                f"{_BASE}/v1/ticker",
                params={"markets": ",".join(chunk)},
                timeout=10,
            ).json()
        def _f(v) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        rows = []
        for d in data:
            mk = d.get("market", "")
            if d.get("trade_price") is None:  # 거래 미형성 신규 종목 제외
                continue
            rows.append({
                "market": mk,
                "코인": names.get(mk, mk),
                "심볼": mk.replace("KRW-", ""),
                "현재가": _f(d.get("trade_price")),
                "등락률": _f(d.get("signed_change_rate")) * 100,
                "거래대금24h": _f(d.get("acc_trade_price_24h")),
            })
        return rows
    except Exception:
        return []


def _b64url(b: bytes) -> bytes:
    return base64.urlsafe_b64encode(b).rstrip(b"=")


def _jwt(params: dict | None = None) -> str | None:
    from urllib.parse import urlencode

    ak = os.getenv("BITHUMB_ACCESS_KEY")
    sk = os.getenv("BITHUMB_SECRET_KEY")
    if not ak or not sk:
        return None
    claims = {
        "access_key": ak,
        "nonce": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
    }
    if params:  # 파라미터 있는 요청은 query_hash 서명 필요
        claims["query_hash"] = hashlib.sha512(urlencode(params).encode()).hexdigest()
        claims["query_hash_alg"] = "SHA512"
    header = _b64url(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64url(json.dumps(claims).encode())
    signing = header + b"." + payload
    sig = _b64url(hmac.new(sk.encode(), signing, hashlib.sha256).digest())
    return (signing + b"." + sig).decode()


def place_order(
    market: str,
    side: str,
    ord_type: str,
    volume: str | None = None,
    price: str | None = None,
    identifier: str | None = None,
) -> dict:
    """빗썸 주문. side: bid(매수)/ask(매도), ord_type: limit/price(시장가매수)/market(시장가매도).

    반환: {"ok": bool, "status": int, "data": dict}
    """
    params = {"market": market, "side": side, "ord_type": ord_type}
    if volume is not None:
        params["volume"] = str(volume)
    if price is not None:
        params["price"] = str(price)
    if identifier:
        params["identifier"] = str(identifier)
    tok = _jwt(params)
    if not tok:
        return {"ok": False, "status": 0, "data": {"error": "API 키 미설정"}}
    try:
        r = requests.post(
            f"{_BASE}/v1/orders",
            json=params,
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            timeout=5,
        )
    except requests.RequestException as exc:
        return {"ok": False, "status": 0, "data": {"error": str(exc)}}
    try:
        data = r.json()
    except Exception:
        data = {"error": r.text[:200]}
    return {"ok": r.ok, "status": r.status_code, "data": data}


def get_asset_total() -> dict | None:
    """빗썸 자산 요약: KRW 예수금 + 코인 평가액 + 합계 + 종목별. 키 없거나 실패 시 None."""
    try:
        tok = _jwt()
        if not tok:
            return None
        accts = requests.get(
            f"{_BASE}/v1/accounts",
            headers={"Authorization": f"Bearer {tok}"},
            timeout=5,
        ).json()
        prices = {r["market"]: r["현재가"] for r in get_krw_tickers()}
        krw_cash = 0.0
        coin_value = 0.0
        items = []
        for a in accts:
            cur = a.get("currency", "")
            bal = float(a.get("balance") or 0) + float(a.get("locked") or 0)
            if bal <= 0:
                continue
            if cur == "KRW":
                krw_cash += bal
                continue
            px = prices.get(f"KRW-{cur}", 0)
            val = bal * px
            coin_value += val
            if val > 0:
                items.append({"코인": cur, "수량": bal, "평가액": val})
        items.sort(key=lambda x: x["평가액"], reverse=True)
        return {
            "krw_cash": krw_cash,
            "coin_value": coin_value,
            "total": krw_cash + coin_value,
            "items": items,
        }
    except Exception:
        return None


if __name__ == "__main__":
    rows = get_krw_tickers()
    assert rows, "티커 조회 실패"
    top = sorted(rows, key=lambda r: r["거래대금24h"], reverse=True)[:5]
    print(f"KRW 마켓 {len(rows)}종목 · 거래대금 TOP 5:")
    for r in top:
        print(f"  {r['코인']:<10} {r['현재가']:>15,.2f}원  {r['등락률']:+.2f}%")
