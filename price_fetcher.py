import requests
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}


def fetch_price(ticker: str) -> dict | None:
    """네이버 금융에서 종목 현재가 정보를 스크래핑한다. 실패 시 None 반환."""
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        # 바이트를 넘겨 BS4가 HTML meta charset을 직접 파싱하게 한다.
        soup = BeautifulSoup(resp.content, "html.parser")

        # ── 현재가 / 방향 ──────────────────────────────────────
        # 네이버는 span.blind 안에 실제 값을 넣고, 시각적 숫자 표시는 별도 span으로 분리한다.
        today_em = soup.select_one("p.no_today em")
        if today_em is None:
            return None

        classes = today_em.get("class", [])
        if "no_up" in classes:
            direction = "up"
        elif "no_down" in classes:
            direction = "down"
        else:
            direction = "same"

        price_blind = today_em.select_one("span.blind")
        price = price_blind.get_text(strip=True) if price_blind else ""

        # ── 전일 대비 / 등락률 ──────────────────────────────────
        exday_ems = soup.select("p.no_exday em")
        change = ""
        rate   = ""

        if exday_ems:
            b = exday_ems[0].select_one("span.blind")
            change = b.get_text(strip=True) if b else ""

        if len(exday_ems) > 1:
            b = exday_ems[1].select_one("span.blind")
            rate = b.get_text(strip=True) if b else ""

        # ── 거래량: dl.blind > dd[거래량 ...] ─────────────────
        volume = ""
        blind_dl = soup.select_one("dl.blind")
        if blind_dl:
            for dd in blind_dl.select("dd"):
                text = dd.get_text(strip=True)
                if text.startswith("거래량"):
                    volume = text.removeprefix("거래량").strip()
                    break

        return {
            "price":      price,
            "change":     change,
            "rate":       rate,
            "volume":     volume,
            "direction":  direction,
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
        }

    except Exception:
        return None
