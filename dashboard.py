"""
dashboard.py — 한경 금융 뉴스 Streamlit 대시보드

실행:
  streamlit run dashboard.py
"""

import atexit
from datetime import datetime
import hashlib
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

import pandas as pd
import streamlit as st
# import yaml  # watchlist_stocks.yaml 로딩 비활성화

import html as _html

from classifier import HybridClassifier
from hankyung_feed import fetch_hankyung_finance
from market_data import load_market_data, market_snapshot_key
from naver_finance_feed import fetch_naver_finance_news, fetch_naver_stock_news
from scorer import score_candidates
from topic_cluster import get_top_topics, get_topic_cluster_status
from trade_signal import analyze_market_state
from web_safety import escape_html, safe_http_url
from tradingview_signals import (
    format_event_time,
    format_price,
    get_latest_signal_for_symbol,
    reason_to_korean,
    signal_label,
    start_webhook_server,
)
# from price_fetcher import fetch_price  # 관심종목 주가 비활성화

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
KST = ZoneInfo("Asia/Seoul")
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
    '<span style="background:#00D9A3;color:#0E1117;border-radius:4px;'
    'padding:1px 7px;font-size:0.72rem;font-weight:600;margin-left:6px;">키워드</span>'
)
BADGE_LLM = (
    '<span style="background:#4B8BFF;color:#0E1117;border-radius:4px;'
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
def cached_topic_cluster_status() -> dict[str, bool]:
    return get_topic_cluster_status()


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
def cached_market_state(ticker: str) -> dict | None:
    return analyze_market_state(ticker)


@st.cache_resource(show_spinner=False)
def ensure_tradingview_webhook_server():
    return start_webhook_server()


@st.cache_data(ttl=15, show_spinner=False)
def cached_latest_tradingview_signal(ticker: str) -> dict | None:
    return get_latest_signal_for_symbol(ticker)


@st.cache_data(ttl=15, show_spinner=False)
def cached_toss_price(ticker: str) -> float | None:
    """토스증권 실시간 현재가 (15초 캐시). 키 미설정/실패 시 None."""
    from toss_client import get_realtime_price
    return get_realtime_price(ticker)


@st.cache_data(ttl=30, show_spinner=False)
def cached_crypto_tickers() -> list[dict]:
    """빗썸 KRW 마켓 코인 시세 (30초 캐시). 실패 시 빈 리스트."""
    from bithumb_client import get_krw_tickers
    return get_krw_tickers()


@st.cache_data(ttl=30, show_spinner=False)
def cached_stock_assets() -> dict | None:
    """토스 주식 자산 (30초 캐시)."""
    from toss_client import get_stock_assets
    return get_stock_assets()


@st.cache_data(ttl=30, show_spinner=False)
def cached_stock_accounts() -> list[dict]:
    """토스 주문 계좌 목록(30초 캐시)."""
    from toss_client import get_stock_accounts
    return get_stock_accounts()


@st.cache_data(ttl=30, show_spinner=False)
def cached_crypto_assets() -> dict | None:
    """빗썸 코인 자산 (30초 캐시)."""
    from bithumb_client import get_asset_total
    return get_asset_total()


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
    safe_sector = escape_html(sector)
    return (
        f'<span style="background:#2A2E3D;color:#FAFAFA;border-radius:4px;'
        f'padding:1px 7px;font-size:0.72rem;margin-left:4px;">{safe_sector}</span>'
    )



# ── 페이지 설정 ───────────────────────────────────────────────

st.set_page_config(
    page_title="국내 주식 마켓 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 비밀번호 잠금 ─────────────────────────────────────────────
# ponytail: 단일 공용 비밀번호(.streamlit/secrets.toml). 사용자별 계정이 필요해지면 streamlit-authenticator로 교체.
import hmac as _hmac


def _check_password() -> bool:
    if st.session_state.get("_authed"):
        return True

    def _verify():
        if _hmac.compare_digest(st.session_state.get("_pw", ""), st.secrets["password"]):
            st.session_state["_authed"] = True
            del st.session_state["_pw"]

    st.text_input("🔒 비밀번호", type="password", key="_pw", on_change=_verify)
    if "_pw" in st.session_state and st.session_state["_pw"]:
        st.error("비밀번호가 틀렸습니다.")
    return False


if not _check_password():
    st.stop()

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

/* ── coing · premium dark ──────────────────────────────────── */
:root {
  --mint:#00E0A8; --mint-soft:#4DEEC4; --gold:#E8C069;
  --ink:#EAEEF5; --muted:#8A93A6;
  --glass:rgba(255,255,255,0.04); --glass-2:rgba(255,255,255,0.06);
  --hair:rgba(255,255,255,0.09); --hair-strong:rgba(255,255,255,0.16);
}
html, body, .stApp, [class*="css"] {
  font-family:'Pretendard','Pretendard Variable',-apple-system,BlinkMacSystemFont,sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 620px at 18% -8%, rgba(0,224,168,0.10), transparent 60%),
    radial-gradient(900px 560px at 100% 0%, rgba(120,110,255,0.08), transparent 55%),
    #090B10;
}
.block-container { padding-top:2.4rem; padding-bottom:3rem; max-width:1400px; }

/* 타이포 */
h1, h2, h3 { letter-spacing:-0.025em; color:var(--ink) !important; font-weight:700; }
h1 {
  font-size:2rem; font-weight:800;
  background:linear-gradient(92deg,#fff 10%,var(--mint) 130%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
h3 { font-size:1.12rem; }
p, span, label, li { color:var(--ink); }

/* 메트릭 = 글래스 카드 */
[data-testid="stMetric"] {
  position:relative; overflow:hidden;
  background:var(--glass);
  border:1px solid var(--hair);
  border-radius:16px; padding:18px 20px;
  backdrop-filter:blur(10px);
  box-shadow:0 8px 28px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
  transition:transform .18s ease, border-color .18s ease;
}
[data-testid="stMetric"]::before {
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--mint),transparent 70%);
  opacity:.8;
}
[data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:var(--hair-strong); }
[data-testid="stMetricValue"] { font-size:1.55rem; font-weight:700; color:#fff; letter-spacing:-0.02em; }
[data-testid="stMetricLabel"] p { color:var(--muted); font-size:0.8rem; font-weight:500; letter-spacing:0.02em; }

/* 테두리 컨테이너 = 글래스 패널 */
[data-testid="stVerticalBlockBorderWrapper"] {
  background:linear-gradient(180deg,var(--glass-2),var(--glass));
  border:1px solid var(--hair) !important;
  border-radius:18px; padding:6px 4px;
  backdrop-filter:blur(12px);
  box-shadow:0 10px 34px rgba(0,0,0,0.32), inset 0 1px 0 rgba(255,255,255,0.04);
}

/* 버튼 */
.stButton > button {
  border-radius:12px; font-weight:600; letter-spacing:-0.01em;
  border:1px solid var(--hair); background:var(--glass); color:var(--ink);
  transition:all .16s ease;
}
.stButton > button:hover { border-color:var(--mint); color:#fff; box-shadow:0 0 0 1px rgba(0,224,168,0.35); }
.stButton > button[kind="primary"] {
  background:linear-gradient(120deg,var(--mint),#00B98C);
  color:#04120D; border:none;
  box-shadow:0 6px 20px rgba(0,224,168,0.28);
}
.stButton > button[kind="primary"]:hover { filter:brightness(1.07); box-shadow:0 8px 26px rgba(0,224,168,0.42); }

/* 탭 */
[data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--hair); }
[data-baseweb="tab"] { font-weight:600; color:var(--muted); }
[data-baseweb="tab"][aria-selected="true"] { color:var(--mint); }
[data-baseweb="tab-highlight"] { background:var(--mint) !important; height:3px; border-radius:3px; }

/* 입력/셀렉트 */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"] {
  border-radius:12px !important;
}

/* 데이터프레임 */
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid var(--hair); }

/* 커스텀 카드 */
.summary-card {
  background:var(--glass); border:1px solid var(--hair);
  border-left:3px solid var(--mint);
  border-radius:16px; padding:22px 26px; margin-bottom:22px;
  color:var(--ink); font-size:1.05rem; line-height:1.9;
  backdrop-filter:blur(10px);
  box-shadow:0 8px 28px rgba(0,0,0,0.3);
}
.summary-card .card-title {
  font-size:0.95rem; font-weight:700; color:var(--mint);
  margin-bottom:14px; letter-spacing:0.06em; text-transform:uppercase;
}
.news-item { padding:11px 0; border-bottom:1px solid var(--hair); }
.news-time { font-size:0.76rem; color:var(--muted); margin-bottom:3px; letter-spacing:0.02em; }
.news-link a { font-size:0.97rem; color:var(--mint-soft); text-decoration:none; font-weight:500; }
.news-link a:hover { color:#fff; }
.reason-item { color:var(--mint-soft); font-size:0.9rem; margin:3px 0; }
.risk-item { color:var(--gold); font-size:0.9rem; margin:3px 0; }
</style>
""", unsafe_allow_html=True)


# ── 사이드바 ① 뉴스 소스 (수집 전에 먼저 정의) ─────────────────

# 사이드바 위젯 대신 기본값 고정 (좌측 UI 정리)
auto_refresh = False
news_source = "네이버 금융 - 전체"
naver_ticker = ""


# ── 뉴스 수집·분류 (버튼 실행 — 초기 로딩 단축) ────────────────
# 뉴스 크롤링·섹터분류·임베딩은 무거우므로 "뉴스 불러오기"를 눌렀을 때만 실행.

_load_news = st.session_state.get("load_news", False)

if _load_news:
    with st.spinner("뉴스 수집 중..."):
        all_articles = load_articles(source=news_source, ticker=naver_ticker)

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
                progress.progress(
                    (i + 1) / len(all_articles),
                    text=f"섹터 분류 중... ({i + 1}/{len(all_articles)})",
                )
            classifier.save_cache()
            progress.empty()
        st.session_state.classified_articles = classified
        st.session_state.classified_source = _articles_key
    classified_articles: list[dict] = st.session_state.get("classified_articles", [])
else:
    classified_articles = []

sector_counts: dict[str, int] = {}
for a in classified_articles:
    sector_counts[a["sector"]] = sector_counts.get(a["sector"], 0) + 1


# ── 사이드바 ② 섹터 필터 및 기타 설정 ───────────────────────

# 요약·모델은 기본값 고정 (좌측 UI 정리)
model = DEFAULT_MODEL
summarize_on = True

with st.sidebar:
    if sector_counts:
        st.subheader("📂 섹터 필터")
        sorted_sectors = sorted(sector_counts.keys())
        sector_labels = ["전체"] + [f"{s} ({sector_counts[s]})" for s in sorted_sectors]
        selected_label: str = st.selectbox("섹터", sector_labels, label_visibility="collapsed")
        selected_sector = (
            "전체" if selected_label == "전체" else selected_label.split(" (")[0]
        )
    else:
        selected_sector = "전체"

    if st.button("🔄 새로고침", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state.pop("classified_articles", None)
        st.session_state.pop("classified_source", None)
        st.session_state.pop("candidates", None)
        st.session_state.pop("candidates_key", None)
        st.session_state.pop("candidates_analyzed_at", None)
        st.rerun()


# 자동 새로고침 — 사이드바 블록 바깥에서 호출해야 정상 동작
if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="price_refresh")


_tv_webhook = ensure_tradingview_webhook_server()
if not _tv_webhook.running:
    st.caption(f"TradingView Webhook 서버를 시작하지 못했습니다: {_tv_webhook.message}")


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


# ── 시장 요약 메트릭 행 (상단) ───────────────────────────────

if _market_df is not None:
    with st.container(border=True):
        _ov_up  = int((_market_df["등락률"] > 0).sum())
        _ov_dn  = int((_market_df["등락률"] < 0).sum())
        _ov_avg = _market_df["등락률"].mean()
        _ov_amt = _market_df["거래대금"].sum() / 1e12  # 조원
        _ov1, _ov2, _ov3, _ov4 = st.columns(4)
        _ov1.metric("📈 상승", f"{_ov_up:,}종목")
        _ov2.metric("📉 하락", f"{_ov_dn:,}종목")
        _ov3.metric("평균 등락률", f"{_ov_avg:+.2f}%")
        _ov4.metric("총 거래대금", f"{_ov_amt:,.1f}조원")


# ── 💰 내 자산 (토스 주식 + 빗썸 코인) ───────────────────────

with st.container(border=True):
    _sa = cached_stock_assets()
    _ca = cached_crypto_assets()
    _stock_total = (_sa["holdings_value"] + _sa["cash"]) if _sa else 0.0
    _crypto_total = _ca["total"] if _ca else 0.0
    _cash = (_sa["cash"] if _sa else 0) + (_ca["krw_cash"] if _ca else 0)
    _grand = _stock_total + _crypto_total
    _base = _grand or 1
    _sp = _stock_total / _base * 100
    _cp = _crypto_total / _base * 100

    st.markdown(f"""
<div style="padding:8px 10px 2px;">
  <div style="font-size:0.72rem;letter-spacing:0.2em;color:var(--muted);font-weight:600;">TOTAL ASSETS · 총 보유자산</div>
  <div style="font-size:2.9rem;font-weight:800;letter-spacing:-0.03em;line-height:1.1;margin:8px 0 2px;
              background:linear-gradient(96deg,#ffffff,var(--mint));-webkit-background-clip:text;
              background-clip:text;-webkit-text-fill-color:transparent;">
     {_grand:,.0f}<span style="font-size:1.15rem;color:var(--muted);-webkit-text-fill-color:var(--muted);margin-left:5px;">원</span>
  </div>
  <div style="display:flex;height:8px;border-radius:6px;overflow:hidden;margin:16px 0 10px;background:rgba(255,255,255,0.06);">
     <div style="width:{_sp:.1f}%;background:linear-gradient(90deg,#4DA3FF,#2E7BE0);"></div>
     <div style="width:{_cp:.1f}%;background:linear-gradient(90deg,var(--mint),#00B98C);"></div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:20px;font-size:0.84rem;color:var(--muted);">
     <span><span style="color:#4DA3FF;">●</span>&nbsp; 토스 주식 &nbsp;<b style="color:var(--ink);">{_stock_total:,.0f}원</b> ({_sp:.0f}%)</span>
     <span><span style="color:var(--mint);">●</span>&nbsp; 빗썸 코인 &nbsp;<b style="color:var(--ink);">{_crypto_total:,.0f}원</b> ({_cp:.0f}%)</span>
     <span><span style="color:var(--gold);">●</span>&nbsp; 현금성 &nbsp;<b style="color:var(--ink);">{_cash:,.0f}원</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    _hh = ('display:flex;justify-content:space-between;padding:8px 2px;'
           'border-bottom:1px solid var(--hair);font-size:0.92rem;')
    _title = ('font-size:0.78rem;color:var(--muted);font-weight:600;'
              'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;')
    _hold_cols = st.columns(2)
    with _hold_cols[0]:
        st.markdown(f'<div style="{_title}">📈 보유 주식</div>', unsafe_allow_html=True)
        if _sa and _sa["items"]:
            for it in _sa["items"]:
                _amt = float(((it.get("marketValue") or {}).get("amount")) or 0)
                _stock_name = _html.escape(str(it.get("name", it.get("symbol", ""))))
                st.markdown(
                    f'<div style="{_hh}"><span>{_stock_name}</span>'
                    f'<span style="font-weight:600;color:#fff;">{_amt:,.0f}원</span></div>',
                    unsafe_allow_html=True)
        else:
            st.caption("보유 주식 없음")
    with _hold_cols[1]:
        st.markdown(f'<div style="{_title}">🪙 보유 코인</div>', unsafe_allow_html=True)
        if _ca and _ca["items"]:
            for it in _ca["items"]:
                _coin_name = _html.escape(str(it.get("코인", "")))
                st.markdown(
                    f'<div style="{_hh}"><span>{_coin_name}</span>'
                    f'<span style="font-weight:600;color:#fff;">{it["평가액"]:,.0f}원</span></div>',
                    unsafe_allow_html=True)
        else:
            st.caption("보유 코인 없음")

    if not _sa and not _ca:
        st.info("자산을 불러오려면 토스·빗썸 API 키가 필요합니다.")


# ── 🔎 검색 & 주문 (실제 체결 — 안전장치 포함) ────────────────

def _fmt_won(x: float) -> str:
    return f"{x:,.0f}원" if x >= 100 else f"{x:,.4f}원"


st.subheader("🔎 검색 & 주문")
with st.container(border=True):
    st.warning(
        "⚠️ API 키에 거래 권한이 있으면 실제 주문이 접수될 수 있습니다. "
        "주문 계좌·종목·방향·수량·가격을 확인한 뒤 거래 잠금과 확인 체크를 켜세요. "
        "지정가 주문의 체결 여부는 증권사·거래소 앱에서 별도로 확인해야 합니다."
    )
    _unlocked = False

    _mkt = st.radio("시장", ["주식 (토스)", "코인 (빗썸)"], horizontal=True, key="trade_mkt")
    _q = st.text_input("종목·코인 검색 (이름 또는 코드)", key="trade_q",
                       placeholder="예: 삼성전자 / 005930 / 비트코인 / BTC")

    _sel = None  # (표시명, 심볼, 현재가, 종류)
    if _q:
        if _mkt.startswith("주식"):
            if _market_df is not None:
                _m = _market_df[
                    _market_df["종목명"].str.contains(_q, case=False, na=False)
                    | _market_df["티커"].astype(str).str.contains(_q, na=False)
                ].head(20)
                if len(_m):
                    _opts = {
                        f"{r['종목명']} ({r['티커']}) — {int(r['현재가']):,}원":
                            (r["종목명"], str(r["티커"]), float(r["현재가"]))
                        for _, r in _m.iterrows()
                    }
                    _pick = st.selectbox("검색 결과", list(_opts), key="trade_pick_s")
                    _name, _sym, _px = _opts[_pick]
                    _sel = (_name, _sym, cached_toss_price(_sym) or _px, "stock")
                else:
                    st.caption("검색 결과가 없습니다.")
        else:
            _cm = [
                c for c in cached_crypto_tickers()
                if _q.lower() in c["코인"].lower() or _q.lower() in c["심볼"].lower()
            ][:20]
            if _cm:
                _opts = {f"{c['코인']} ({c['심볼']}) — {_fmt_won(c['현재가'])}": c for c in _cm}
                _pick = st.selectbox("검색 결과", list(_opts), key="trade_pick_c")
                _c = _opts[_pick]
                _sel = (_c["코인"], _c["market"], _c["현재가"], "coin")
            else:
                st.caption("검색 결과가 없습니다.")

    if _sel:
        _name, _sym, _px, _kind = _sel
        _safe_name = _html.escape(str(_name))
        st.markdown(f"### {_safe_name}  ·  현재가 {_fmt_won(_px)}")
        _side_kr = st.radio("주문 방향", ["매수", "매도"], horizontal=True, key="trade_side")
        _account_seq = None
        _account_label = "빗썸 API 키 연결 계정"

        if _kind == "stock":
            _accounts = cached_stock_accounts()
            if _accounts:
                _account_options = {}
                for _account in _accounts:
                    _seq = _account.get("accountSeq")
                    if _seq is None:
                        continue
                    _account_name = str(
                        _account.get("accountName")
                        or _account.get("name")
                        or "토스증권 계좌"
                    )
                    _account_number = str(_account.get("accountNumber") or "")
                    _suffix = f" · 끝 {_account_number[-4:]}" if _account_number else ""
                    _label = f"{_account_name}{_suffix} · ID {_seq}"
                    _account_options[_label] = _seq
                if _account_options:
                    _account_label = st.selectbox(
                        "주문 계좌",
                        list(_account_options),
                        key="trade_account_s",
                    )
                    _account_seq = _account_options[_account_label]
                else:
                    st.error("주문 가능한 토스 계좌 식별 정보를 찾지 못했습니다.")
            else:
                st.error("토스 주문 계좌를 불러오지 못했습니다. API 설정을 확인하세요.")
            _qty = st.number_input("수량 (주)", min_value=1, step=1, value=1, key="trade_qty_s")
            _price = st.number_input("지정가 (원)", min_value=0, value=int(_px), step=10, key="trade_price_s")
            _est = _qty * _price
        else:
            _qty = st.number_input("수량 (코인)", min_value=0.0, value=0.0,
                                   step=0.0001, format="%.8f", key="trade_qty_c")
            _price = st.number_input("지정가 (원)", min_value=0.0, value=float(_px),
                                     step=1.0, format="%.4f", key="trade_price_c")
            _est = _qty * _price

        _safe_account_label = escape_html(_account_label)
        st.markdown(
            f"**계좌** {_safe_account_label} · **{_side_kr}** · {_safe_name} · "
            f"수량 **{_qty:g}** · 지정가 **{_fmt_won(_price)}** "
            f"→ 예상 금액 **{_est:,.0f}원** (수수료 별도)"
        )
        _min_order_ok = _kind != "coin" or _est >= 5000
        _account_ok = _kind != "stock" or _account_seq is not None
        if not _min_order_ok and _est > 0:
            st.error("빗썸 최소 주문금액은 5,000원입니다.")

        _order_day = datetime.now(KST).date().isoformat()
        _order_payload = (
            f"{_order_day}|{_kind}|{_account_seq or 'bithumb'}|{_sym}|"
            f"{_side_kr}|{_qty:.8f}|{_price:.8f}"
        )
        _order_sig = hashlib.sha256(_order_payload.encode("utf-8")).hexdigest()[:32]
        _unlocked = st.checkbox("🔓 이 주문의 거래 잠금 해제", value=False, key=f"trade_unlock_{_order_sig}")
        _confirm = st.checkbox("위 주문 내용을 확인했습니다", key=f"trade_confirm_{_order_sig}")
        _already_attempted = st.session_state.get("last_attempted_order") == _order_sig
        if _already_attempted:
            st.info(
                "같은 날짜·계좌·종목·방향·수량·가격의 주문을 이미 전송했습니다. "
                "접수·체결 상태를 증권사 또는 거래소 앱에서 확인하세요."
            )
        _ready = (
            _unlocked
            and _confirm
            and _est > 0
            and _min_order_ok
            and _account_ok
            and not _already_attempted
        )
        if st.button("🚀 주문 실행", type="primary", disabled=not _ready, key="trade_go"):
            st.session_state["last_attempted_order"] = _order_sig
            with st.spinner("주문 전송 중..."):
                if _kind == "stock":
                    from toss_client import place_stock_order
                    _res = place_stock_order(
                        _sym,
                        "BUY" if _side_kr == "매수" else "SELL",
                        int(_qty),
                        _price,
                        "LIMIT",
                        account_seq=_account_seq,
                        client_order_id=_order_sig,
                    )
                else:
                    from bithumb_client import place_order
                    _res = place_order(
                        _sym,
                        "bid" if _side_kr == "매수" else "ask",
                        "limit",
                        volume=_qty,
                        price=int(_price) if _price >= 1 else _price,
                        identifier=_order_sig,
                    )
            if _res["ok"]:
                st.success(f"✅ 주문 접수 응답: {_res['data']}")
                st.cache_data.clear()  # 자산·잔고 갱신
            elif _res["status"] == 0:
                st.warning(
                    f"⚠️ 주문 결과를 확인할 수 없습니다: {_res['data']}\n\n"
                    "네트워크 오류일 수 있으므로 재전송하지 말고 증권사·거래소 앱에서 접수 여부를 확인하세요."
                )
            else:
                st.error(f"❌ 주문 거절 응답 (status {_res['status']}): {_res['data']}")

    if not _unlocked:
        st.caption("🔒 현재 거래 잠금 상태 — 검색·조회는 되지만 주문은 실행되지 않습니다.")


# ── 후보 종목 (그리드, 항상 표시) ───────────────────────────

st.subheader("🎯 오늘의 단기 관찰 종목 TOP 10")
st.caption("거래대금·시가총액 필터 → 등락률 기반 1차 점수 → 일봉 거래량·MA20·RSI(14) 2차 보강")

if "selected_candidate" not in st.session_state:
    st.session_state["selected_candidate"] = None

_cand_ready = st.session_state.get("show_candidates") or "candidates" in st.session_state
if _market_df is not None and not _cand_ready:
    if st.button("📊 관찰 종목 분석 실행 (일봉 조회 ~30초)", type="primary", key="run_cand"):
        st.session_state["show_candidates"] = True
        st.rerun()
    st.caption("버튼을 누르면 오늘의 단기 관찰 종목 TOP 10을 분석합니다.")
elif _market_df is not None:
    _candidates_key = market_snapshot_key(_market_df, _market_date)
    if (
        "candidates" not in st.session_state
        or st.session_state.get("candidates_key") != _candidates_key
    ):
        with st.spinner("후보 종목 분석 중... (일봉 데이터 최대 30종목 조회)"):
            st.session_state["candidates"] = score_candidates(
                _market_df,
                snapshot_date=_market_date,
            )
            st.session_state["candidates_key"] = _candidates_key
            st.session_state["candidates_analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    candidates = st.session_state["candidates"]

    _analyzed_at = st.session_state.get("candidates_analyzed_at", "-")
    st.caption(f"후보 분석 시각: {_analyzed_at} · 시장 기준 거래일: {_market_date or '확인 불가'}")

    if not candidates:
        st.info(
            "조건을 통과한 후보 종목이 없습니다. "
            "오늘은 거래대금·시가총액 필터와 1차 점수를 충족한 종목이 없을 수 있습니다."
        )
    else:
        _stage2_done = candidates[0].get("2단계완료수", 0)
        _stage2_total = candidates[0].get("2단계대상수", 0)
        if _stage2_total and _stage2_done == 0:
            st.error(
                "일봉 2단계 분석을 완료하지 못했습니다(데이터 없음·부족 또는 조회 오류). "
                "현재 목록은 1단계 스냅샷 점수만 반영합니다."
            )
        elif _stage2_done < _stage2_total:
            st.warning(
                f"일봉 2단계 분석 일부 미완료: {_stage2_done}/{_stage2_total}종목 완료. "
                "표시된 카드의 상세 화면에서 해당 종목 상태를 확인할 수 있습니다."
            )

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
            change_pct = sel["등락률"]
            if change_pct > 0:
                change_color, change_arrow = "#FF4B5C", "▲"
            elif change_pct < 0:
                change_color, change_arrow = "#4B8BFF", "▼"
            else:
                change_color, change_arrow = "#8B92A6", "–"
            with st.container(border=True):
                st.markdown(f"### {sel['종목명']}")
                _d1, _d2, _d3, _d4 = st.columns(4)
                _d1.metric("현재가", f"{sel['현재가']:,}원")
                _d2.markdown(
                    '<div style="font-size:0.8rem;color:#8B92A6;">등락률</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:{change_color};">'
                    f"{change_arrow} {abs(change_pct):.2f}%</div>",
                    unsafe_allow_html=True,
                )
                _d3.metric("거래대금", f"{sel['거래대금(억원)']:,}억원")
                _d4.markdown(
                    '<div style="font-size:0.8rem;color:#8B92A6;">점수</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:#00D9A3;">{sel["점수"]}</div>',
                    unsafe_allow_html=True,
                )

            _stage2_status = sel.get("2단계상태", "상태 없음")
            _bars_date = sel.get("일봉기준일") or "-"
            _observation_type = sel.get("관찰유형", "조건 충족 관찰")
            if sel.get("2단계완료"):
                st.success(f"2단계 일봉 분석 완료 · 일봉 기준일 {_bars_date} · {_observation_type}")
            else:
                _stage2_error = sel.get("2단계오류") or "일봉 데이터가 없습니다."
                st.warning(f"2단계 {_stage2_status} · {_stage2_error} · 현재 점수는 1단계 중심")
            if sel.get("기준일불일치"):
                st.warning(
                    f"데이터 기준일 불일치: 시장 스냅샷 {_market_date or '-'} / 일봉 {_bars_date}"
                )
            if sel.get("미완성봉가능"):
                _collected_date = sel.get("수집일봉기준일") or "-"
                st.info(
                    f"장중 미완성 최신봉({_collected_date})을 모든 기술지표에서 제외하고 "
                    f"{_bars_date} 완성봉까지 분석했습니다."
                )

            # ── 토스증권 실시간 시세 (FDR 지연 시세 대비) ──────────
            _toss_price = cached_toss_price(sel_ticker)
            if _toss_price:
                _fdr = float(sel["현재가"])
                _gap = (_toss_price - _fdr) / _fdr * 100 if _fdr else 0
                st.markdown(
                    f'<div style="background:#1A1D29;border:1px solid #2A2E3D;border-left:4px solid #4B8BFF;'
                    f'border-radius:12px;padding:10px 16px;margin-bottom:10px;">'
                    f'<span style="color:#4B8BFF;font-weight:700;">⚡ 토스증권 실시간</span> &nbsp; '
                    f'<span style="color:#FAFAFA;font-size:1.25rem;font-weight:700;">{_toss_price:,.0f}원</span> &nbsp; '
                    f'<span style="color:#8B92A6;font-size:0.82rem;">FDR 지연 시세 {_fdr:,.0f}원 대비 {_gap:+.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("⚡ 토스 실시간 시세: 조회 불가 (API 키 미설정이거나 장 시간 외)")

            st.markdown("#### 📡 최신 기술 신호")
            tv_signal = cached_latest_tradingview_signal(sel_ticker)
            if tv_signal is None:
                st.info("현재 이 종목에 수신된 TradingView 신호가 없습니다.")
            elif tv_signal.get("is_expired"):
                st.info(
                    "⚪ 신호 만료\n\n"
                    "마지막 신호가 유효 시간 범위를 지났습니다."
                )
            else:
                normalized_signal = tv_signal["signal_normalized"]
                label, label_color = signal_label(normalized_signal)
                if normalized_signal == "NEUTRAL":
                    st.info(
                        "⚪ 중립\n\n"
                        "현재 유효한 기술적 신호가 없습니다.\n\n"
                        f"발생 시각: {format_event_time(tv_signal['event_time_utc'])}\n\n"
                        f"시간봉: {tv_signal.get('timeframe') or '-'}분\n\n"
                        "출처: TradingView"
                    )
                else:
                    with st.container(border=True):
                        st.markdown(
                            f'<div style="font-weight:700;color:{label_color};font-size:1.05rem;">'
                            f'{label}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(reason_to_korean(tv_signal.get("reason")))
                        st.markdown(f"발생 가격: {format_price(tv_signal.get('price'))}")
                        st.markdown(f"시간봉: {tv_signal.get('timeframe') or '-'}분")
                        st.markdown(f"발생 시각: {format_event_time(tv_signal['event_time_utc'])}")
                        st.markdown("출처: TradingView")

            col_r, col_w = st.columns(2)
            with col_r:
                for r in sel["후보이유"]:
                    score_str = f' <span style="color:#00D9A3;font-weight:700;">+{r[1]}</span>' if r[1] > 0 else ""
                    st.markdown(
                        f'<div class="reason-item">✅ {r[0]}{score_str}</div>',
                        unsafe_allow_html=True,
                    )
            with col_w:
                for r in sel["주의점"]:
                    score_str = f' <span style="color:#FFB020;font-weight:700;">{r[1]}</span>' if r[1] != 0 else ""
                    st.markdown(
                        f'<div class="risk-item">⚠️ {r[0]}{score_str}</div>',
                        unsafe_allow_html=True,
                    )

            # ── 단기 변동성 관찰 차트 + 시장 상태 ───────────────
            st.markdown("#### ⏱️ 단기 변동성 관찰")
            with st.spinner("지표 상태 분석 중..."):
                timing = cached_market_state(sel_ticker)

            if timing is None:
                st.info("일봉 데이터가 부족해 지표 상태를 계산할 수 없습니다.")
            else:
                _sig = timing["state"]
                _sig_color = {"상방 조건 우세": "#FF4B5C", "하방 리스크 우세": "#4B8BFF"}.get(_sig, "#2A2E3D")
                st.markdown(
                    f'<span style="background:{_sig_color};color:#fff;border-radius:20px;'
                    f'padding:3px 14px;font-size:0.9rem;font-weight:700;">'
                    f'현재 시장 상태: {_sig}</span>',
                    unsafe_allow_html=True,
                )
                for _r in timing["reasons"]:
                    st.markdown(f"- {_r}")

                _lv = timing["levels"]
                _lv_cols = st.columns(3)
                _lv_cols[0].metric("🛡️ 하단 기준선 (20일 저가)", f"{_lv['하단 기준선(20일 저가)']:,.0f}원",
                                   help="하단 기준 이탈 주의 구간")
                _lv_cols[1].metric("📏 MA20 (단기 추세 기준선)", f"{_lv['MA20']:,.0f}원",
                                   help="중심선 근접 구간")
                _lv_cols[2].metric("🎯 상단 기준선 (60일 고가)", f"{_lv['상단 기준선(60일 고가)']:,.0f}원",
                                   help="상단 기준 근접 구간")

                import plotly.graph_objects as go

                _bars = timing["bars"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=_bars.index, y=_bars["Close"], name="종가",
                                         line=dict(color="#FAFAFA", width=1.6)))
                fig.add_trace(go.Scatter(x=_bars.index, y=_bars["MA5"], name="MA5",
                                         line=dict(color="#FFC857", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=_bars.index, y=_bars["MA20"], name="MA20",
                                         line=dict(color="#00D9A3", width=1)))

                _ups   = [e for e in timing["events"] if e["side"] == "상방"]
                _downs = [e for e in timing["events"] if e["side"] == "하방"]
                if _ups:
                    fig.add_trace(go.Scatter(
                        x=[e["date"] for e in _ups], y=[e["price"] for e in _ups],
                        mode="markers", name="상방 전환 이벤트",
                        marker=dict(symbol="triangle-up", size=12, color="#FF4B5C"),
                        text=[e["label"] for e in _ups],
                        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>%{y:,.0f}원<extra>상방</extra>",
                    ))
                if _downs:
                    fig.add_trace(go.Scatter(
                        x=[e["date"] for e in _downs], y=[e["price"] for e in _downs],
                        mode="markers", name="하방 전환 이벤트",
                        marker=dict(symbol="triangle-down", size=12, color="#4B8BFF"),
                        text=[e["label"] for e in _downs],
                        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>%{y:,.0f}원<extra>하방</extra>",
                    ))

                fig.add_hline(y=_lv["하단 기준선(20일 저가)"], line_dash="dash",
                              line_color="#00D9A3", line_width=1,
                              annotation_text="하단 기준선", annotation_position="bottom left")
                fig.add_hline(y=_lv["상단 기준선(60일 고가)"], line_dash="dash",
                              line_color="#FFB020", line_width=1,
                              annotation_text="상단 기준선", annotation_position="top left")

                fig.update_layout(
                    height=380, margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    xaxis_title=None, yaxis_title=None,
                    hovermode="x unified",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#1A1D29",
                    font=dict(color="#FAFAFA", family="monospace"),
                    xaxis=dict(gridcolor="#2A2E3D", showgrid=True),
                    yaxis=dict(gridcolor="#2A2E3D", showgrid=True),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "▲ 상방 전환 이벤트: MA5가 MA20 상향 돌파·RSI 30 상향 통과 / "
                    "▼ 하방 전환 이벤트: MA5가 MA20 하향 이탈·RSI 70 하향 통과 — "
                    "과거 조건 발생 지점의 시각화이며 매매 신호가 아닙니다."
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
    st.subheader("📊 시장 데이터")
    with st.container(border=True):
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
                    safe_name = escape_html(row["종목명"])
                    st.markdown(
                        f"**{safe_name}** &nbsp; {int(row['현재가']):,}원 &nbsp; "
                        f'<span style="color:#FF4B5C;font-weight:700;">▲ {pct:.2f}%</span>',
                        unsafe_allow_html=True,
                    )

            with col_dn:
                st.markdown("**하락률 TOP 5**")
                for _, row in top_dn.iterrows():
                    pct = row["등락률"]
                    safe_name = escape_html(row["종목명"])
                    st.markdown(
                        f"**{safe_name}** &nbsp; {int(row['현재가']):,}원 &nbsp; "
                        f'<span style="color:#4B8BFF;font-weight:700;">▼ {abs(pct):.2f}%</span>',
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


# ── 🪙 코인 시장 (빗썸) ───────────────────────────────────────

st.subheader("🪙 코인 시장 (빗썸)")
with st.container(border=True):
    _crypto = cached_crypto_tickers()
    if not _crypto:
        st.info("빗썸 코인 시세를 불러오지 못했습니다. 잠시 후 새로고침해 보세요.")
    else:
        _cdf = pd.DataFrame(_crypto)
        _tab_hot, _tab_price = st.tabs(["🔥 인기 (24h 거래대금)", "💵 가격순"])

        def _render_coins(_df: pd.DataFrame) -> None:
            _t = _df.head(10).copy()
            _t.insert(0, "순위", range(1, len(_t) + 1))
            _t["현재가"] = _t["현재가"].map(lambda x: f"{x:,.0f}원" if x >= 100 else f"{x:,.2f}원")
            _t["등락률"] = _t["등락률"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
            _t["거래대금(억)"] = (_df.head(10)["거래대금24h"] / 1e8).round(0).astype(int).values
            st.dataframe(
                _t[["순위", "코인", "심볼", "현재가", "등락률", "거래대금(억)"]],
                use_container_width=True, hide_index=True,
            )

        with _tab_hot:
            _render_coins(_cdf.sort_values("거래대금24h", ascending=False))
        with _tab_price:
            _render_coins(_cdf.sort_values("현재가", ascending=False))

    st.caption("빗썸 KRW 마켓 · 공개 시세 · 30초 캐시")

st.divider()


# ── 📰 뉴스 & 요약 + 인기 주제 (버튼 실행 — 초기 로딩 단축) ─────

if not _load_news:
    st.subheader("📰 뉴스 & 요약")
    if st.button("📰 뉴스 불러오기 (수집·분류·요약·인기주제)", type="primary", key="run_news"):
        st.session_state["load_news"] = True
        st.rerun()
    st.caption("네이버 금융 뉴스 수집 → 섹터분류 → 요약 → 인기주제. 초기 로딩 단축을 위해 버튼 실행 방식입니다.")
else:
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
                    f"<div>• {escape_html(line.strip())}</div>"
                    for line in summary.splitlines()
                    if line.strip()
                )
                safe_sector = escape_html(selected_sector)
                st.markdown(f"""
<div class="summary-card">
  <div class="card-title">📰 오늘의 핵심 흐름 — {safe_sector}</div>
  {lines}
</div>
""", unsafe_allow_html=True)
            else:
                st.warning("LLM 요약에 실패했습니다 (ANTHROPIC_API_KEY 확인).")
        elif summarize_on and not articles:
            st.info(f"'{selected_sector}' 관련 기사가 없어 요약을 생성하지 않았습니다.")

    with col_right:
        st.subheader("📌 인기 주제 TOP 3")
        st.caption("임베딩 의미 유사도와 기업·기관 엔티티 그래프를 결합해 관련 주제를 추립니다.")
        if not articles:
            st.info("기사가 없어 주제 분석을 건너뜁니다.")
        else:
            with st.spinner("주제 클러스터링 중..."):
                topics = cached_top_topics(tuple(a.get("title", "") for a in articles))

            if not topics:
                status = cached_topic_cluster_status()
                missing = [
                    name
                    for name, installed in (
                        ("scikit-learn", status["scikit_learn"]),
                    )
                    if not installed
                ]
                if missing:
                    st.info(
                        f"`{', '.join(missing)}` 패키지가 설치되지 않았습니다.\n\n"
                        "```\npip install scikit-learn\n```"
                    )
                else:
                    st.info(
                        "주제 분석 결과가 비어 있습니다. 기사 수가 적거나 서로 비슷한 헤드라인이 부족할 수 있습니다."
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
                        others_html += f" &nbsp;<span style='color:#8B92A6;'>외 {overflow}건</span>"

                    rep = _html.escape(topic["rep_title"])
                    match_method = escape_html(topic.get("match_method", "하이브리드 매칭"))
                    st.markdown(f"""
<div style="border-left:4px solid #00D9A3;padding:12px 18px;margin-bottom:10px;
            background:#1A1D29;border:1px solid #2A2E3D;border-radius:12px;">
  <div style="font-weight:700;font-size:0.88rem;color:#8B92A6;margin-bottom:4px;">
    {int(topic['rank'])}위 · {int(topic['count'])}건 · {match_method}
  </div>
  <div style="font-size:1.0rem;color:#FAFAFA;font-weight:600;margin-bottom:{'6px' if others_html else '0'};">
    📌 {rep}
  </div>
  {'<div style="font-size:0.82rem;color:#8B92A6;line-height:1.8;">' + others_html + '</div>' if others_html else ''}
</div>
""", unsafe_allow_html=True)

    st.divider()


# ── 뉴스 헤드라인 (접힘) ──────────────────────────────────────

if _load_news:
  with st.expander(f"📋 뉴스 헤드라인 ({len(articles)}건)", expanded=False):
    if not articles:
        st.info("해당 섹터의 기사가 없습니다. 다른 섹터를 선택해 보세요.")
    else:
        show_sector_tag = selected_sector == "전체"

        for article in articles:
            title    = escape_html(article.get("title", ""))
            link     = safe_http_url(article.get("link", ""))
            pub      = article.get("published", "")
            sector   = article.get("sector", "기타")
            method   = article.get("method", "")
            press    = escape_html(article.get("press", ""))
            time_str = escape_html(str(pub)[:25]) if pub else ""

            badge = BADGE_KEYWORD if method == "키워드" else BADGE_LLM
            stag  = sector_tag_html(sector) if show_sector_tag else ""
            press_badge = (
                f'<span style="background:rgba(0,217,163,0.12);color:#00D9A3;border-radius:4px;'
                f'padding:1px 7px;font-size:0.72rem;font-weight:600;margin-right:4px;">'
                f'{press}</span>'
                if press else ""
            )

            st.markdown(f"""
<div class="news-item">
  <div class="news-time">{time_str}</div>
  <div class="news-link">{press_badge}<a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>{stag}{badge}</div>
</div>
""", unsafe_allow_html=True)


# ── 데이터 기준일 (하단) ──────────────────────────────────────

if _market_df is not None:
    _snapshot_at = _market_df.attrs.get("snapshot_at", "-")
    _market_date_label = _market_date or "확인 불가"
    st.caption(
        f"📅 시장 기준 거래일: {_market_date_label} · 스냅샷 조회 시각: {_snapshot_at} · "
        "시장 스냅샷 5분 캐시, 변동 시 관찰 종목 재분석"
    )
else:
    st.caption("📅 시장 데이터를 불러오지 못했습니다.")

st.caption(
    "ℹ️ 이 대시보드는 공개 시장 데이터와 사용자가 정의한 기술적 조건을 시각화하는 학습용 도구입니다. "
    "특정 종목의 매수·매도 추천, 수익 예측, 자동매매 기능을 제공하지 않습니다."
)
st.caption("기술적 지표 기반 신호이며 실제 매수·매도 권유가 아닙니다.")
