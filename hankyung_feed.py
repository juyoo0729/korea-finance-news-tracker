import feedparser
import sys

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def fetch_hankyung_finance(max_items: int = 10) -> list[dict]:
    try:
        feed = feedparser.parse(
            "https://www.hankyung.com/feed/finance",
            request_headers=HEADERS,
        )
    except ValueError:
        feed = feedparser.parse(
            "https://www.hankyung.com/feed/finance",
        )

    status = feed.get("status")
    if status is not None and status != 200:
        raise RuntimeError(f"피드 요청 실패: HTTP {status}")

    results = []
    for entry in feed.entries[:max_items]:
        link = entry.get("link", "") or ""
        if link and not link.startswith("http"):
            link = ""
        results.append({
            "title":     entry.get("title", ""),
            "published": entry.get("published", ""),
            "link":      link,
            "summary":   entry.get("summary", "")[:100] if entry.get("summary") else "",
            "press":     "한국경제",
        })
    return results


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    articles = fetch_hankyung_finance(max_items=10)
    print(f"한국경제 금융 뉴스 — 최신 {len(articles)}건\n" + "=" * 60)
    for i, a in enumerate(articles, 1):
        print(f"\n[{i}] {a['title']}")
        print(f"    날짜  : {a['published']}")
        print(f"    링크  : {a['link']}")
        if a["summary"]:
            print(f"    요약  : {a['summary']}...")
