"""
dashboard.py — 한경 금융 뉴스 Streamlit 대시보드

실행:
  streamlit run dashboard.py
"""

import atexit
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

import pandas as pd
import streamlit as st
# import yaml  # watchlist_stocks.yaml 로딩 비활성화

import html as _html

from classifier import HybridClassifier
from hankyung_feed import fetch_hankyung_finance
from market_data import load_market_data
from naver_finance_feed import fetch_naver_finance_news, fetch_naver_stock_news
from scorer import score_candidates
from topic_cluster import get_top_topics
from trade_signal import analyze_trade_timing
# from price_fetcher import fetch_price  # 관심종목 주가 비활성화

DEFAULT_MODEL = "gpt-5.4-mini"
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

@st.cache_data(ttl=300, show_spinner=False)
def load_articles(max_items: int = 50, source: str = "한경 RSS", ticker: str = "") -> list[dict]:
    if source == "네이버 금융 - 전체":
        raw = fetch_naver_finance_news(max_items=max_items)
    elif source == "네이버 금융 - 종목별":
        raw = fetch_naver_stock_news(ticker=ticker, max_items=max_items) if ticker else []
    else:
        raw = fetch_hankyung_finance(max_items=max_items)
    return _deduplicate(raw)


@st.cache_data(ttl=300, show_spinner=False)
def cached_top_topics(titles: tuple[str, ...]) -> list[dict]:
    return get_top_topics(list(titles))


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_backtest() -> tuple[pd.DataFrame, str] | None:
    """results/ 폴더에서 가장 최근 백테스트 CSV를 읽는다. 없으면 None."""
    files = sorted(
        Path(__file__).parent.glob("results/backtest_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return None
    try:
        df = pd.read_csv(files[-1], dtype={"티커": str})
        return df, files[-1].name
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def cached_trade_timing(ticker: str) -> dict | None:
    return analyze_trade_timing(ticker)


@st.cache_data(show_spinner=False)
def get_summary(headlines: tuple[str, ...], keyword: str, model: str) -> str | None:
    try:
        from llm_client import _generate
        prompt = SUMMARY_PROMPT.format(
            headlines="\n".join(f"- {h}" for h in headlines),
            keyword=keyword,
        )
        return _generate(prompt, model=model or None)
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


# ── 사이드바 ① 뉴스 소스 (수집 전에 먼저 정의) ─────────────────

with st.sidebar:
    st.title("⚙️ 설정")
    auto_refresh = st.checkbox("🔄 자동 새로고침 (60초)", value=False)
    st.divider()
    st.subheader("📡 뉴스 소스")
    news_source = "네이버 금융 - 전체"
    naver_ticker = ""
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

    # 뉴스 메트릭 — 섹터 필터 바로 아래
    _filtered_for_metrics = (
        classified_articles if selected_sector == "전체"
        else [a for a in classified_articles if a["sector"] == selected_sector]
    )
    st.metric("검색 결과", f"{len(_filtered_for_metrics)}건",
              delta=f"전체 {len(classified_articles)}건 중")
    if _filtered_for_metrics:
        _latest = _filtered_for_metrics[0].get("published", "")
        st.metric("최신 기사", _latest[:16] if _latest else "-")

    st.divider()
    model = st.text_input("🤖 모델", value=DEFAULT_MODEL)
    summarize_on = st.toggle("요약 생성", value=True)
    st.divider()

    if st.button("🔄 새로고침", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state.pop("classified_articles", None)
        st.session_state.pop("classified_source", None)
        st.session_state.pop("candidates", None)
        st.session_state.pop("candidates_key", None)
        st.rerun()

    st.caption("크롤링: 네이버 금융 전체")
    st.divider()
    with st.expander("DEBUG"):
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


# ── 시장 데이터 로딩 ─────────────────────────────────────────

with st.spinner("시장 데이터 로딩 중..."):
    _market_df, _market_date = load_market_data()


# ═══════════════════ 메인 화면 ═══════════════════════════════

st.title("📈 국내 주식 마켓 대시보드")


# ── 후보 종목 (그리드, 항상 표시) ───────────────────────────

st.subheader("🎯 오늘의 후보 종목 TOP 10")
st.caption("1단계: 등락률·거래대금 절대 기준 → 상위 30개 / 2단계: 일봉 거래량·MA20·RSI(14) 보강")

if "selected_candidate" not in st.session_state:
    st.session_state["selected_candidate"] = None

if _market_df is not None:
    _candidates_key = str(_market_date or "")
    if (
        "candidates" not in st.session_state
        or st.session_state.get("candidates_key") != _candidates_key
    ):
        with st.spinner("후보 종목 분석 중... (일봉 데이터 최대 30종목 조회)"):
            st.session_state["candidates"]     = score_candidates(_market_df)
            st.session_state["candidates_key"] = _candidates_key
    candidates = st.session_state["candidates"]

    if not candidates:
        st.info(
            "조건을 통과한 후보 종목이 없습니다. "
            "오늘은 거래대금·등락률 기준을 충족한 종목이 없을 수 있습니다."
        )
    else:
        # ── 5열 × 2행 버튼 그리드 ──────────────────────────────
        grid = st.columns(5)
        for i, c in enumerate(candidates):
            with grid[i % 5]:
                is_selected = st.session_state["selected_candidate"] == c["티커"]
                if st.button(
                    f"#{i + 1} {c['종목명']}\n점수 {c['점수']}",
                    key=f"cand_{c['티커']}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state["selected_candidate"] = c["티커"]

        # ── 선택 종목 상세 ──────────────────────────────────────
        sel_ticker = st.session_state["selected_candidate"]
        sel = next((c for c in candidates if c["티커"] == sel_ticker), None)

        if sel is None:
            st.caption("카드를 클릭하면 상세가 표시됩니다")
        else:
            change_pct   = sel["등락률"]
            change_color = "#e03131" if change_pct >= 0 else "#1971c2"
            change_arrow = "▲" if change_pct >= 0 else "▼"
            st.markdown(
                f"**{sel['종목명']}** &nbsp; {sel['현재가']:,}원 &nbsp; "
                f'<span style="color:{change_color};font-weight:700;">'
                f"{change_arrow} {abs(change_pct):.2f}%</span> &nbsp; "
                f"거래대금 {sel['거래대금(억원)']:,}억원 &nbsp; "
                f'<span class="candidate-score">점수 {sel["점수"]}</span>',
                unsafe_allow_html=True,
            )

            col_r, col_w = st.columns(2)
            with col_r:
                for r in sel["후보이유"]:
                    score_str = f' <span style="color:#2e7d32;font-weight:700;">+{r[1]}</span>' if r[1] > 0 else ""
                    st.markdown(
                        f'<div class="reason-item">✅ {r[0]}{score_str}</div>',
                        unsafe_allow_html=True,
                    )
            with col_w:
                for r in sel["주의점"]:
                    score_str = f' <span style="color:#e65100;font-weight:700;">{r[1]}</span>' if r[1] != 0 else ""
                    st.markdown(
                        f'<div class="risk-item">⚠️ {r[0]}{score_str}</div>',
                        unsafe_allow_html=True,
                    )

            # ── 매매 타이밍 차트 + 신호 ─────────────────────────
            st.markdown("#### ⏱️ 매매 타이밍")
            with st.spinner("매매 신호 분석 중..."):
                timing = cached_trade_timing(sel_ticker)

            if timing is None:
                st.info("일봉 데이터가 부족해 매매 신호를 계산할 수 없습니다.")
            else:
                _sig = timing["signal"]
                _sig_color = {"매수 우위": "#e03131", "매도 우위": "#1971c2"}.get(_sig, "#6c757d")
                st.markdown(
                    f'<span style="background:{_sig_color};color:#fff;border-radius:20px;'
                    f'padding:3px 14px;font-size:0.9rem;font-weight:700;">'
                    f'현재 신호: {_sig}</span>',
                    unsafe_allow_html=True,
                )
                for _r in timing["reasons"]:
                    st.markdown(f"- {_r}")

                _lv = timing["levels"]
                _lv_cols = st.columns(3)
                _lv_cols[0].metric("🛡️ 지지선 (20일 저가)", f"{_lv['지지선(20일 저가)']:,.0f}원",
                                   help="이탈 시 손절 검토 구간")
                _lv_cols[1].metric("📏 MA20 (눌림목)", f"{_lv['MA20']:,.0f}원",
                                   help="조정 시 분할 매수 참고 구간")
                _lv_cols[2].metric("🎯 저항선 (60일 고가)", f"{_lv['저항선(60일 고가)']:,.0f}원",
                                   help="도달 시 차익실현 검토 구간")

                import plotly.graph_objects as go

                _bars = timing["bars"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=_bars.index, y=_bars["Close"], name="종가",
                                         line=dict(color="#222", width=1.6)))
                fig.add_trace(go.Scatter(x=_bars.index, y=_bars["MA5"], name="MA5",
                                         line=dict(color="#f0a500", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=_bars.index, y=_bars["MA20"], name="MA20",
                                         line=dict(color="#1a73e8", width=1)))

                _buys  = [e for e in timing["events"] if e["side"] == "매수"]
                _sells = [e for e in timing["events"] if e["side"] == "매도"]
                if _buys:
                    fig.add_trace(go.Scatter(
                        x=[e["date"] for e in _buys], y=[e["price"] for e in _buys],
                        mode="markers", name="매수 신호",
                        marker=dict(symbol="triangle-up", size=12, color="#e03131"),
                        text=[e["label"] for e in _buys],
                        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>%{y:,.0f}원<extra>매수</extra>",
                    ))
                if _sells:
                    fig.add_trace(go.Scatter(
                        x=[e["date"] for e in _sells], y=[e["price"] for e in _sells],
                        mode="markers", name="매도 신호",
                        marker=dict(symbol="triangle-down", size=12, color="#1971c2"),
                        text=[e["label"] for e in _sells],
                        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>%{y:,.0f}원<extra>매도</extra>",
                    ))

                fig.add_hline(y=_lv["지지선(20일 저가)"], line_dash="dash",
                              line_color="#2e7d32", line_width=1,
                              annotation_text="지지선", annotation_position="bottom left")
                fig.add_hline(y=_lv["저항선(60일 고가)"], line_dash="dash",
                              line_color="#e65100", line_width=1,
                              annotation_text="저항선", annotation_position="top left")

                fig.update_layout(
                    height=380, margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    xaxis_title=None, yaxis_title=None,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "▲ 매수 신호: 골든크로스(MA5>MA20)·RSI 30 상향 탈출 / "
                    "▼ 매도 신호: 데드크로스·RSI 70 하향 이탈 — 참고용이며 투자 판단의 책임은 본인에게 있습니다."
                )
else:
    st.warning("시장 데이터를 불러오지 못했습니다.")


# ── 백테스트 결과 ─────────────────────────────────────────────

with st.expander("🧪 백테스트 결과 — 스코어링 룰의 과거 성과", expanded=False):
    _bt = load_latest_backtest()
    if _bt is None:
        st.info(
            "백테스트 결과가 아직 없습니다. 터미널에서 실행 후 새로고침하세요:\n\n"
            "```\npython backtest.py --start 2026-01-01 --end 2026-05-31 --interval 5\n```"
        )
    else:
        _bt_df, _bt_name = _bt
        from backtest import summarize_by_score, summarize_overall

        _overall = summarize_overall(_bt_df)
        st.caption(
            f"파일: `{_bt_name}` · 기준일 {_bt_df['기준일'].min()} ~ {_bt_df['기준일'].max()} · "
            f"후보 {len(_bt_df)}건 (바닥반등 {int(_bt_df['바닥반등'].sum())}건)"
        )

        _metric_cols = st.columns(len(_overall))
        for _col, (_, _row) in zip(_metric_cols, _overall.iterrows()):
            with _col:
                if _row["표본"] == 0 or pd.isna(_row["후보 평균(%)"]):
                    st.metric(_row["구간"], "표본 없음")
                else:
                    st.metric(
                        f"{_row['구간']} 평균 수익률",
                        f"{_row['후보 평균(%)']:+.2f}%",
                        delta=f"{_row['초과수익(%p)']:+.2f}%p vs KOSPI · 승률 {_row['승률(%)']:.0f}%",
                    )

        st.markdown("**구간별 상세 (vs KOSPI)**")
        st.dataframe(_overall, use_container_width=True, hide_index=True)

        st.markdown("**점수 구간별 — 고득점 후보가 실제로 나았는지**")
        st.dataframe(summarize_by_score(_bt_df), use_container_width=True, hide_index=True)

        st.caption(
            "※ 유니버스는 현재 거래대금 상위 종목 기준이라 생존 편향이 일부 있으며, "
            "거래대금·시가총액은 일봉 근사값입니다. 자세한 한계는 backtest.py 참고."
        )

st.divider()


# ── 시장 데이터 탭 ────────────────────────────────────────────

if _market_df is not None:
    tab_up, tab_amount, tab_cap = st.tabs(
        ["🔥 등락률 TOP", "💰 거래대금 TOP 10", "🏦 시가총액 TOP 10"]
    )

    with tab_up:
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

    with tab_amount:
        top_amount = _market_df.nlargest(10, "거래대금")[["종목명", "거래대금", "등락률"]].copy()
        top_amount.insert(0, "순위", range(1, 11))
        top_amount["거래대금(억원)"] = (top_amount["거래대금"] / 1e8).round(0).astype(int)
        top_amount["등락률"] = top_amount["등락률"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
        top_amount = top_amount[["순위", "종목명", "거래대금(억원)", "등락률"]].reset_index(drop=True)
        st.dataframe(top_amount, use_container_width=True, hide_index=True)

    with tab_cap:
        top_cap = _market_df.nlargest(10, "시가총액")[["종목명", "시가총액", "현재가", "등락률"]].copy()
        top_cap.insert(0, "순위", range(1, 11))
        top_cap["시총(조원)"] = (top_cap["시가총액"] / 1e12).round(2)
        top_cap["현재가"] = top_cap["현재가"].astype(int).map(lambda x: f"{x:,}원")
        top_cap["등락률"] = top_cap["등락률"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
        top_cap = top_cap[["순위", "종목명", "시총(조원)", "현재가", "등락률"]].reset_index(drop=True)
        st.dataframe(top_cap, use_container_width=True, hide_index=True)

    st.divider()


# ── LLM 요약 + 인기 주제 (좌우 배치) ─────────────────────────

col_left, col_right = st.columns([0.6, 0.4])

with col_left:
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
            st.warning("LLM 요약에 실패했습니다. 백엔드 설정과 모델이 올바른지 확인하세요.")
    elif summarize_on and not articles:
        st.info(f"'{selected_sector}' 관련 기사가 없어 요약을 생성하지 않았습니다.")

with col_right:
    st.subheader("📌 인기 주제 TOP 3")
    st.caption("sentence-transformers 임베딩으로 유사 헤드라인을 묶어 가장 많이 다뤄진 주제를 추립니다.")
    if not articles:
        st.info("기사가 없어 주제 분석을 건너뜁니다.")
    else:
        with st.spinner("주제 클러스터링 중..."):
            topics = cached_top_topics(tuple(a.get("title", "") for a in articles))

        if not topics:
            st.info(
                "`sentence-transformers` 또는 `scikit-learn`이 설치되지 않았습니다.\n\n"
                "```\npip install sentence-transformers scikit-learn\n```"
            )
        else:
            for topic in topics:
                others = [h for h in topic["headlines"] if h != topic["rep_title"]]
                others_preview = others[:4]
                overflow = len(others) - len(others_preview)
                others_html = " &nbsp;/&nbsp; ".join(
                    _html.escape(h) for h in others_preview
                )
                if overflow > 0:
                    others_html += f" &nbsp;<span style='color:#aaa;'>외 {overflow}건</span>"

                rep = _html.escape(topic["rep_title"])
                st.markdown(f"""
<div style="border-left:4px solid #f0a500;padding:12px 18px;margin-bottom:10px;
            background:#fffdf5;border-radius:6px;">
  <div style="font-weight:700;font-size:0.88rem;color:#888;margin-bottom:4px;">
    {topic['rank']}위 · {topic['count']}건
  </div>
  <div style="font-size:1.0rem;color:#222;font-weight:600;margin-bottom:{'6px' if others_html else '0'};">
    📌 {rep}
  </div>
  {'<div style="font-size:0.82rem;color:#999;line-height:1.8;">' + others_html + '</div>' if others_html else ''}
</div>
""", unsafe_allow_html=True)

st.divider()


# ── 뉴스 헤드라인 (접힘) ──────────────────────────────────────

with st.expander(f"📋 뉴스 헤드라인 ({len(articles)}건)", expanded=False):
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

if _market_df is not None and _market_date:
    st.caption(f"📅 시장 데이터 기준일: {_market_date} · 5분마다 자동 갱신 (FinanceDataReader)")
else:
    st.caption("📅 시장 데이터를 불러오지 못했습니다.")
