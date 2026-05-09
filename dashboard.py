"""
dashboard.py — 한경 금융 뉴스 Streamlit 대시보드

실행:
  streamlit run dashboard.py
"""

import atexit
from pathlib import Path

import streamlit as st
import yaml

from classifier import HybridClassifier
from hankyung_feed import fetch_hankyung_finance
from naver_finance_feed import fetch_naver_finance_news, fetch_naver_stock_news
from price_fetcher import fetch_price

DEFAULT_MODEL = "exaone3.5:2.4b"
COLS_PER_ROW = 4

SUMMARY_PROMPT = """\
[지시사항]
- 출력 언어: 한국어만 사용. 영어, 한자, 외래어 사용 금지.
- 형식: 세 줄. 각 줄은 완결된 문장 하나. 번호·기호 없음.
- 근거: 아래 헤드라인에 있는 내용만 사용. 없는 내용 추가 금지.

[헤드라인]
{headlines}

[요청]
위 헤드라인에서 '{keyword}' 관련 핵심 흐름을 한국어로 세 줄 요약하라."""

BADGE_KEYWORD = (
    '<span style="background:#28a745;color:#fff;border-radius:4px;'
    'padding:1px 7px;font-size:0.72rem;font-weight:600;margin-left:6px;">키워드</span>'
)
BADGE_LLM = (
    '<span style="background:#1a73e8;color:#fff;border-radius:4px;'
    'padding:1px 7px;font-size:0.72rem;font-weight:600;margin-left:6px;">LLM</span>'
)


def _deduplicate(articles: list[dict]) -> list[dict]:
    """link URL 기준 중복 제거."""
    seen: set[str] = set()
    result: list[dict] = []
    for item in articles:
        url = item.get("link", "")
        if url and url not in seen:
            seen.add(url)
            result.append(item)
        elif not url:
            result.append(item)
    return result


# ── 데이터 함수 ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_watchlist() -> dict[str, str]:
    yaml_path = Path(__file__).parent / "watchlist_stocks.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # YAML이 숫자 키를 int로 파싱할 경우 대비해 str로 변환
    return {str(k).zfill(6): v for k, v in raw.items()}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_prices(tickers: tuple[str, ...]) -> dict[str, dict | None]:
    return {ticker: fetch_price(ticker) for ticker in tickers}


@st.cache_data(ttl=300, show_spinner=False)
def load_articles(max_items: int = 50, source: str = "한경 RSS", ticker: str = "") -> list[dict]:
    if source == "네이버 금융 - 전체":
        raw = fetch_naver_finance_news(max_items=max_items)
    elif source == "네이버 금융 - 종목별":
        raw = fetch_naver_stock_news(ticker=ticker, max_items=max_items) if ticker else []
    else:
        raw = fetch_hankyung_finance(max_items=max_items)
    return _deduplicate(raw)


@st.cache_data(show_spinner=False)
def get_summary(headlines: tuple[str, ...], keyword: str, model: str) -> str | None:
    try:
        import ollama
        prompt = SUMMARY_PROMPT.format(
            headlines="\n".join(f"- {h}" for h in headlines),
            keyword=keyword,
        )
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"].strip()
    except Exception:
        return None


def sector_tag_html(sector: str) -> str:
    return (
        f'<span style="background:#6c757d;color:#fff;border-radius:4px;'
        f'padding:1px 7px;font-size:0.72rem;margin-left:4px;">{sector}</span>'
    )


# ── 페이지 설정 ───────────────────────────────────────────────

st.set_page_config(
    page_title="한경 금융 뉴스 대시보드",
    page_icon="📰",
    layout="wide",
)

st.markdown("""
<style>
.summary-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 5px solid #f0a500;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 24px;
    color: #f0f0f0;
    font-size: 1.08rem;
    line-height: 1.9;
}
.summary-card .card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0a500;
    margin-bottom: 14px;
    letter-spacing: 0.03em;
}
.news-item {
    padding: 10px 0;
    border-bottom: 1px solid #e8e8e8;
}
.news-time {
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 3px;
}
.news-link a {
    font-size: 0.98rem;
    color: #1a73e8;
    text-decoration: none;
    font-weight: 500;
}
.news-link a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)


# ── 타이틀 ───────────────────────────────────────────────────

st.title("📰 한경 금융 뉴스 대시보드")


# ── 관심 종목 주가 ────────────────────────────────────────────

st.subheader("📈 관심 종목")

watchlist = load_watchlist()
tickers = tuple(watchlist.keys())

with st.spinner("주가 수집 중..."):
    prices = fetch_all_prices(tickers)

ticker_items = list(watchlist.items())
for row_start in range(0, len(ticker_items), COLS_PER_ROW):
    row = ticker_items[row_start : row_start + COLS_PER_ROW]
    cols = st.columns(COLS_PER_ROW)
    for j, (ticker, name) in enumerate(row):
        data = prices.get(ticker)
        with cols[j]:
            if data:
                sign = (
                    "+" if data["direction"] == "up"
                    else ("-" if data["direction"] == "down" else "")
                )
                delta = f"{sign}{data['rate']}% ({sign}{data['change']}원)"
                st.metric(name, f"{data['price']}원", delta)
            else:
                st.metric(name, "조회 실패", None)

valid_prices = [p for p in prices.values() if p]
if valid_prices:
    st.caption(f"마지막 업데이트: {valid_prices[0]['fetched_at']} · 60초마다 자동 갱신")

st.divider()


# ── 사이드바 ① 뉴스 소스 (수집 전에 먼저 정의) ─────────────────

with st.sidebar:
    st.title("⚙️ 설정")
    auto_refresh = st.checkbox("🔄 자동 새로고침 (60초)", value=False)
    st.divider()
    st.subheader("📡 뉴스 소스")
    news_source = st.selectbox(
        "소스 선택",
        options=["한경 RSS", "네이버 금융 - 전체", "네이버 금융 - 종목별"],
        label_visibility="collapsed",
    )
    naver_ticker = ""
    if news_source == "네이버 금융 - 종목별":
        naver_ticker = st.text_input(
            "종목 코드 (6자리)",
            value="005930",
            placeholder="예: 005930 (삼성전자)",
        ).strip()
    st.divider()


# ── 뉴스 수집 ─────────────────────────────────────────────────

with st.spinner("뉴스 수집 중..."):
    all_articles = load_articles(source=news_source, ticker=naver_ticker)

print(f"[DEBUG] source={news_source!r}, ticker={naver_ticker!r}, count={len(all_articles)}")
if all_articles:
    print(f"[DEBUG] 첫 번째 기사 press={all_articles[0].get('press')!r}, title={all_articles[0].get('title','')[:30]!r}")


# ── 섹터 분류 (세션 캐시) ─────────────────────────────────────
# 소스·종목이 바뀌면 이전 분류 결과를 버리고 재분류한다.

_articles_key = f"{news_source}|{naver_ticker}"
_need_reclassify = (
    "classified_articles" not in st.session_state
    or st.session_state.get("classified_source") != _articles_key
)

if _need_reclassify:
    classifier = HybridClassifier()
    atexit.register(classifier.save_cache)

    classified: list[dict] = []
    if all_articles:
        progress = st.progress(0, text="분류 시작...")
        for i, article in enumerate(all_articles):
            sector, method = classifier.classify(article["title"])
            classified.append({**article, "sector": sector, "method": method})
            label = "키워드 분류 중..." if method == "키워드" else "LLM 분류 중..."
            progress.progress(
                (i + 1) / len(all_articles),
                text=f"{label} ({i + 1}/{len(all_articles)})",
            )
        classifier.save_cache()
        progress.empty()

    st.session_state.classified_articles = classified
    st.session_state.classified_source = _articles_key

classified_articles: list[dict] = st.session_state.classified_articles

stats = {
    "키워드": sum(1 for a in classified_articles if a["method"] == "키워드"),
    "LLM":   sum(1 for a in classified_articles if a["method"] == "LLM"),
}

sector_counts: dict[str, int] = {}
for a in classified_articles:
    s = a["sector"]
    sector_counts[s] = sector_counts.get(s, 0) + 1


# ── 사이드바 ② 섹터 필터 및 기타 설정 ───────────────────────

with st.sidebar:
    st.subheader("📂 섹터 필터")
    sorted_sectors = sorted(sector_counts.keys())
    sector_labels = ["전체"] + [f"{s} ({sector_counts[s]})" for s in sorted_sectors]
    selected_label: str = st.selectbox("섹터", sector_labels, label_visibility="collapsed")
    selected_sector = (
        "전체" if selected_label == "전체" else selected_label.split(" (")[0]
    )

    total = stats["키워드"] + stats["LLM"]
    if total > 0:
        kw_pct  = stats["키워드"] / total * 100
        llm_pct = stats["LLM"]    / total * 100
        st.caption(f"분류 방법: 키워드 {kw_pct:.0f}% / LLM {llm_pct:.0f}%")

    st.divider()

    model = st.text_input("🤖 Ollama 모델", value=DEFAULT_MODEL)
    summarize_on = st.toggle("요약 생성", value=True)

    st.divider()

    if st.button("🔄 새로고침", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state.pop("classified_articles", None)
        st.session_state.pop("classified_source", None)
        st.rerun()

    label_map = {
        "한경 RSS": "RSS: 한국경제신문 금융",
        "네이버 금융 - 전체": "크롤링: 네이버 금융 전체",
        "네이버 금융 - 종목별": f"크롤링: 네이버 금융 [{naver_ticker or '종목 미선택'}]",
    }
    st.caption(label_map.get(news_source, ""))

    st.divider()
    st.write(f"**DEBUG**")
    st.write(f"source: `{news_source}`")
    st.write(f"articles: `{len(classified_articles)}건`")
    if classified_articles:
        first = classified_articles[0]
        st.write(f"press[0]: `{first.get('press', '(없음)')}`")
        st.write(f"title[0]: `{first.get('title', '')[:25]}`")


# 자동 새로고침 — 사이드바 블록 바깥에서 호출해야 정상 동작
if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="price_refresh")


# ── 섹터 필터 적용 ────────────────────────────────────────────

if selected_sector == "전체":
    articles = classified_articles
else:
    articles = [a for a in classified_articles if a["sector"] == selected_sector]


# ── 뉴스 메트릭 ──────────────────────────────────────────────

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("검색 결과", f"{len(articles)}건", delta=f"전체 {len(classified_articles)}건 중")
with col2:
    if articles:
        latest = articles[0].get("published", "")
        st.metric("최신 기사", latest[:16] if latest else "-")

st.divider()


# ── 요약 카드 ─────────────────────────────────────────────────

if summarize_on and articles:
    with st.spinner(f"'{selected_sector}' 요약 생성 중 ({model})..."):
        summary = get_summary(
            tuple(a["title"] for a in articles),
            selected_sector,
            model,
        )

    if summary:
        lines = "".join(
            f"<div>• {line.strip()}</div>"
            for line in summary.splitlines()
            if line.strip()
        )
        st.markdown(f"""
<div class="summary-card">
  <div class="card-title">📰 오늘의 핵심 흐름 — {selected_sector}</div>
  {lines}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning(
            "Ollama에 연결할 수 없어 요약을 건너뜁니다. "
            "`ollama serve` 가 실행 중인지 확인하세요."
        )

elif summarize_on and not articles:
    st.info(f"'{selected_sector}' 관련 기사가 없어 요약을 생성하지 않았습니다.")


# ── 헤드라인 리스트 ───────────────────────────────────────────

st.subheader(f"뉴스 헤드라인 ({len(articles)}건)")

if not articles:
    st.info("해당 섹터의 기사가 없습니다. 다른 섹터를 선택해 보세요.")
else:
    show_sector_tag = selected_sector == "전체"

    for article in articles:
        title    = article.get("title", "")
        link     = article.get("link", "#")
        pub      = article.get("published", "")
        sector   = article.get("sector", "기타")
        method   = article.get("method", "")
        press    = article.get("press", "")
        time_str = pub[:25] if pub else ""

        badge = BADGE_KEYWORD if method == "키워드" else BADGE_LLM
        stag  = sector_tag_html(sector) if show_sector_tag else ""
        press_badge = (
            f'<span style="background:#e8f0fe;color:#1a73e8;border-radius:4px;'
            f'padding:1px 7px;font-size:0.72rem;font-weight:600;margin-right:4px;">'
            f'{press}</span>'
            if press else ""
        )

        st.markdown(f"""
<div class="news-item">
  <div class="news-time">{time_str}</div>
  <div class="news-link">{press_badge}<a href="{link}" target="_blank">{title}</a>{stag}{badge}</div>
</div>
""", unsafe_allow_html=True)
