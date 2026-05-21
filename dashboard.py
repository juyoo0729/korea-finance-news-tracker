"""
dashboard.py — 한경 금융 뉴스 Streamlit 대시보드

실행:
  streamlit run dashboard.py
"""

import atexit
# from pathlib import Path  # watchlist yaml 경로 불필요

import streamlit as st
# import yaml  # watchlist_stocks.yaml 로딩 비활성화

from classifier import HybridClassifier
from hankyung_feed import fetch_hankyung_finance
from market_data import load_market_data
from naver_finance_feed import fetch_naver_finance_news, fetch_naver_stock_news
from scorer import score_candidates
# from price_fetcher import fetch_price  # 관심종목 주가 비활성화

DEFAULT_MODEL = "exaone3.5:2.4b"
# COLS_PER_ROW = 4  # 관심종목 카드 레이아웃 비활성화

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

# load_watchlist / fetch_all_prices — 관심종목 기능 비활성화
# @st.cache_data(show_spinner=False)
# def load_watchlist() -> dict[str, str]: ...
# @st.cache_data(ttl=60, show_spinner=False)
# def fetch_all_prices(tickers): ...

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
    page_title="국내 주식 마켓 대시보드",
    page_icon="📈",
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
.candidate-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-left: 4px solid #1a73e8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.candidate-score {
    display: inline-block;
    background: #1a73e8;
    color: #fff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.85rem;
    font-weight: 700;
}
.reason-item {
    color: #2e7d32;
    font-size: 0.9rem;
    margin: 2px 0;
}
.risk-item {
    color: #e65100;
    font-size: 0.9rem;
    margin: 2px 0;
}
</style>
""", unsafe_allow_html=True)


# ── 타이틀 ───────────────────────────────────────────────────

st.title("📈 국내 주식 마켓 대시보드")


# ── 등락률 TOP 섹션 ───────────────────────────────────────────

with st.spinner("시장 데이터 로딩 중..."):
    _market_df, _market_date = load_market_data()

if _market_df is not None:
    st.subheader("🔥 등락률 TOP 5")
    col_up, col_dn = st.columns(2)

    top_up = _market_df.nlargest(5, "등락률")[["종목명", "현재가", "등락률"]].reset_index(drop=True)
    top_dn = _market_df.nsmallest(5, "등락률")[["종목명", "현재가", "등락률"]].reset_index(drop=True)

    with col_up:
        st.markdown("**상승률 TOP 5**")
        for _, row in top_up.iterrows():
            pct = row["등락률"]
            st.markdown(
                f"**{row['종목명']}** &nbsp; {int(row['현재가']):,}원 &nbsp; "
                f'<span style="color:#e03131;font-weight:700;">▲ {pct:.2f}%</span>',
                unsafe_allow_html=True,
            )

    with col_dn:
        st.markdown("**하락률 TOP 5**")
        for _, row in top_dn.iterrows():
            pct = row["등락률"]
            st.markdown(
                f"**{row['종목명']}** &nbsp; {int(row['현재가']):,}원 &nbsp; "
                f'<span style="color:#1971c2;font-weight:700;">▼ {abs(pct):.2f}%</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 거래대금 TOP 10 ───────────────────────────────────────
    st.subheader("💰 거래대금 TOP 10")

    top_amount = _market_df.nlargest(10, "거래대금")[["종목명", "거래대금", "등락률"]].copy()
    top_amount.insert(0, "순위", range(1, 11))
    top_amount["거래대금(억원)"] = (top_amount["거래대금"] / 1e8).round(0).astype(int)
    top_amount["등락률"] = top_amount["등락률"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    top_amount = top_amount[["순위", "종목명", "거래대금(억원)", "등락률"]].reset_index(drop=True)

    st.dataframe(top_amount, use_container_width=True, hide_index=True)
    st.divider()

    # ── 시가총액 TOP 10 ───────────────────────────────────────
    st.subheader("🏦 시가총액 TOP 10")

    top_cap = _market_df.nlargest(10, "시가총액")[["종목명", "시가총액", "현재가", "등락률"]].copy()
    top_cap.insert(0, "순위", range(1, 11))
    top_cap["시총(조원)"] = (top_cap["시가총액"] / 1e12).round(2)
    top_cap["현재가"] = top_cap["현재가"].astype(int).map(lambda x: f"{x:,}원")
    top_cap["등락률"] = top_cap["등락률"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    top_cap = top_cap[["순위", "종목명", "시총(조원)", "현재가", "등락률"]].reset_index(drop=True)

    st.dataframe(top_cap, use_container_width=True, hide_index=True)
    st.divider()

    # ── 오늘의 후보 종목 ───────────────────────────────────────────
    st.subheader("🎯 오늘의 후보 종목 TOP 10")
    st.caption("1단계: 등락률·거래대금 절대 기준 → 상위 30개 / 2단계: 일봉 거래량·MA20·RSI(14) 보강")

    with st.spinner("후보 종목 분석 중... (일봉 데이터 최대 30종목 조회)"):
        candidates = score_candidates(_market_df)

    if not candidates:
        st.info(
            "조건을 통과한 후보 종목이 없습니다. "
            "오늘은 거래대금·등락률 기준을 충족한 종목이 없을 수 있습니다."
        )
    else:
        for i, c in enumerate(candidates, 1):
            change_pct  = c["등락률"]
            change_color = "#e03131" if change_pct >= 0 else "#1971c2"
            change_arrow = "▲" if change_pct >= 0 else "▼"

            reasons_html = "".join(
                f'<div class="reason-item">✅ {r}</div>' for r in c["후보이유"]
            )
            risks_html = "".join(
                f'<div class="risk-item">⚠️ {r}</div>' for r in c["주의점"]
            )
            detail_html = reasons_html + risks_html

            st.markdown(f"""
<div class="candidate-card">
  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:{'8px' if detail_html else '0'};">
    <span style="font-size:1.05rem;font-weight:700;color:#333;">#{i} {c['종목명']}</span>
    <span style="font-size:0.82rem;color:#777;">{c['티커']}</span>
    <span style="font-size:0.98rem;font-weight:600;">{c['현재가']:,}원</span>
    <span style="font-size:0.95rem;font-weight:700;color:{change_color};">{change_arrow} {abs(change_pct):.2f}%</span>
    <span style="font-size:0.85rem;color:#555;">거래대금 {c['거래대금(억원)']:,}억원</span>
    <span class="candidate-score">점수 {c['점수']}</span>
  </div>
  {detail_html}
</div>
""", unsafe_allow_html=True)

    st.divider()

else:
    st.warning("시장 데이터를 불러오지 못했습니다.")
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


# ── 데이터 기준일 (하단) ──────────────────────────────────────

st.divider()
if _market_df is not None and _market_date:
    st.caption(f"📅 시장 데이터 기준일: {_market_date} · 5분마다 자동 갱신 (FinanceDataReader)")
else:
    st.caption("📅 시장 데이터를 불러오지 못했습니다.")
