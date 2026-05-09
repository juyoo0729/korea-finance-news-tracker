"""네이버 금융 뉴스 수집 모듈"""
import time
import requests
from bs4 import BeautifulSoup, Tag


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_MAIN_NEWS_URL = "https://finance.naver.com/news/mainnews.naver"
_STOCK_NEWS_URL = "https://finance.naver.com/item/news.naver"


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response | None:
    """재시도 로직이 포함된 GET 요청."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            resp.encoding = "euc-kr"
            return resp
        except requests.RequestException as e:
            print(f"[naver] 요청 오류 ({attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(1.5 * attempt)
    return None


def _abs_link(href: str) -> str:
    if not href:
        return ""
    return "https://finance.naver.com" + href if href.startswith("/") else href


# ──────────────────────────────────────────────────────────
# 메인 뉴스 파서
# 실제 구조:
#   ul.recentNewsList > li
#     dl > dd.articleSubject > a (제목/링크)
#          dd.articleSummary    (요약 + span.press + span.wdate)
# ──────────────────────────────────────────────────────────
def _parse_main_item(li: Tag) -> dict | None:
    title_tag = li.select_one("dd.articleSubject a")
    if not title_tag:
        # 폴백: 아무 a 태그
        title_tag = li.select_one("a[href]")
    if not title_tag:
        return None

    title = title_tag.get_text(strip=True)
    if not title:
        return None

    link = _abs_link(title_tag.get("href", ""))

    summary_dd = li.select_one("dd.articleSummary")
    press, published, summary = "", "", ""

    if summary_dd:
        press_tag = summary_dd.select_one("span.press")
        press = press_tag.get_text(strip=True) if press_tag else ""

        time_tag = summary_dd.select_one("span.wdate")
        published = time_tag.get_text(strip=True) if time_tag else ""

        # 요약: span을 제거한 나머지 텍스트
        for s in summary_dd.select("span"):
            s.decompose()
        summary = summary_dd.get_text(strip=True)

    return {"title": title, "link": link, "press": press, "published": published, "summary": summary}


# ──────────────────────────────────────────────────────────
# 종목 뉴스 파서
# 실제 구조:
#   table.type5 > tbody > tr
#     td.title > a (제목/링크)
#     td (언론사)
#     td (날짜)
# ──────────────────────────────────────────────────────────
def _parse_stock_row(tr: Tag) -> dict | None:
    title_tag = tr.select_one("td.title a")
    if not title_tag:
        return None

    title = title_tag.get_text(strip=True)
    if not title:
        return None

    link = _abs_link(title_tag.get("href", ""))

    tds = tr.select("td")
    press = tds[1].get_text(strip=True) if len(tds) > 1 else ""
    published = tds[2].get_text(strip=True) if len(tds) > 2 else ""

    return {"title": title, "link": link, "press": press, "published": published, "summary": ""}


# ──────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────
def fetch_naver_finance_news(max_items: int = 30) -> list[dict]:
    """네이버 금융 메인 뉴스 페이지에서 최신 뉴스를 가져온다."""
    resp = _get(_MAIN_NEWS_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # ul.recentNewsList 우선, 없으면 ul.newsList
    container = soup.select_one("ul.recentNewsList") or soup.select_one("ul.newsList")
    if container is None:
        print("[naver] 뉴스 목록 컨테이너를 찾을 수 없습니다. HTML 구조가 변경됐을 수 있습니다.")
        return []

    articles: list[dict] = []
    for li in container.select("li"):
        if len(articles) >= max_items:
            break
        item = _parse_main_item(li)
        if item:
            articles.append(item)
        time.sleep(0.05)

    return articles


def fetch_naver_stock_news(ticker: str, max_items: int = 20) -> list[dict]:
    """특정 종목 코드의 네이버 금융 뉴스를 가져온다.

    Args:
        ticker: 6자리 종목 코드 (예: '005930' → 삼성전자)
    """
    resp = _get(_STOCK_NEWS_URL, params={"code": ticker})
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.select_one("table.type5")
    if table is None:
        print(f"[naver] 종목({ticker}) 뉴스 테이블을 찾을 수 없습니다.")
        return []

    articles: list[dict] = []
    for tr in table.select("tbody tr"):
        if len(articles) >= max_items:
            break
        item = _parse_stock_row(tr)
        if item:
            articles.append(item)
        time.sleep(0.05)

    return articles
