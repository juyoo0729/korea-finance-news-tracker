"""
backtest.py — scorer.py 스코어링 룰의 과거 성과 검증 (백테스트)

과거 거래일마다 "그날 기준 후보 TOP 10"을 재현하고, 각 후보의 +5/+10/+20거래일
수익률을 같은 기간 KOSPI와 비교한다. 스코어링 로직은 scorer.py를 import해 그대로 재사용한다.

사용법:
  python backtest.py --start 2026-01-01 --end 2026-05-31 --interval 5
  python backtest.py --start 2026-05-10 --end 2026-06-10 --interval 10 --universe 300

설계 메모 (한계 포함):
  - 미래 데이터 누수(look-ahead) 방지: 기준일 D의 스코어링에는 D까지의 일봉만 잘라서
    전달한다. 수익률은 D 종가 → D+N거래일 종가 기준이다.
  - 과거 시점의 전종목 스냅샷(등락률·거래대금·시가총액)을 무료로 받을 방법이 없어
    (pykrx는 KRX 로그인 필수로 변경됨), "현재 상장 종목 중 거래대금 상위 N개"의
    일봉으로 스냅샷을 재구성한다.
      · 거래대금 ≈ 종가 × 거래량        (일봉에 거래대금 컬럼이 없어 근사)
      · 시가총액 ≈ 현재 시가총액 × (당시 종가 / 최신 종가)
    → 유니버스를 현재 기준으로 고르므로 생존 편향(survivorship bias)이 일부 존재한다.
  - FinanceDataReader 호출 결과는 data_cache/ 폴더에 parquet으로 캐싱해 재사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

import FinanceDataReader as fdr

from scorer import CANDLE_LOOKBACK_DAYS, MIN_MARKET_CAP, score_candidates

CACHE_DIR   = Path(__file__).parent / "data_cache"
RESULTS_DIR = Path(__file__).parent / "results"
META_PATH   = CACHE_DIR / "_cache_meta.json"

KOSPI_INDEX   = "KS11"
HORIZONS      = (5, 10, 20)  # 수익률 추적 구간 (거래일)
LOOKBACK_PAD  = 130          # 첫 기준일 이전 확보할 캘린더 일수 (scorer의 90일 + 여유)
FORWARD_PAD   = 45           # 마지막 기준일 이후 확보할 캘린더 일수 (+20거래일 보장)
FETCH_WORKERS = 6            # 동시 일봉 요청 수 — 과도한 동시 요청은 차단 위험
REBOUND_TAG   = "바닥 변곡점"

console = Console(highlight=False)


# ── 일봉 캐시 ─────────────────────────────────────────────────────

class BarsCache:
    """티커별 일봉 parquet 캐시.

    _cache_meta.json에 요청 범위와 수집 시각을 기록해, 캐시가 요청 범위를 덮거나
    오늘 이미 받아온 티커(상장폐지 등으로 최근 봉이 없는 경우 포함)는 재요청하지 않는다.
    """

    def __init__(self):
        CACHE_DIR.mkdir(exist_ok=True)
        try:
            self.meta: dict = json.loads(META_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.meta = {}

    def _path(self, ticker: str) -> Path:
        return CACHE_DIR / f"bars_{ticker}.parquet"

    def _is_valid(self, ticker: str, start: str, end: str) -> bool:
        m = self.meta.get(ticker)
        if not m or not self._path(ticker).exists():
            return False
        today = datetime.today().strftime("%Y-%m-%d")
        return m["start"] <= start and (m["end"] >= end or m.get("fetched_at") == today)

    def get(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        """캐시 또는 fdr에서 [start, end] 일봉을 반환한다. 실패 시 None."""
        if self._is_valid(ticker, start, end):
            try:
                return pd.read_parquet(self._path(ticker))
            except Exception:
                pass  # 파일 손상 → 재요청
        try:
            df = fdr.DataReader(ticker, start, end)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        df.to_parquet(self._path(ticker))
        self.meta[ticker] = {
            "start": start,
            "end": end,
            "fetched_at": datetime.today().strftime("%Y-%m-%d"),
        }
        return df

    def save_meta(self) -> None:
        META_PATH.write_text(
            json.dumps(self.meta, ensure_ascii=False), encoding="utf-8"
        )


def fetch_all_bars(
    cache: BarsCache, tickers: list[str], start: str, end: str
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """유니버스 전체의 일봉을 병렬 수집한다. (성공 dict, 실패 티커 목록) 반환."""
    bars: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = {ex.submit(cache.get, t, start, end): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            df = fut.result()
            if df is None or df.empty:
                failed.append(t)
            else:
                bars[t] = df
            if i % 100 == 0 or i == len(tickers):
                console.print(f"  일봉 수집 {i}/{len(tickers)} (실패 {len(failed)})")

    cache.save_meta()
    return bars, failed


# ── 유니버스·스냅샷 재구성 ────────────────────────────────────────

def load_universe(n: int) -> pd.DataFrame:
    """현재 상장 KOSPI+KOSDAQ 중 거래대금 상위 n개를 백테스트 유니버스로 반환한다.

    주의: 현재 기준으로 고르므로 과거에 활발했다가 지금 한산해진(또는 상장폐지된)
    종목은 빠진다 — 생존 편향. 모듈 docstring 참고.
    """
    df = pd.concat(
        [fdr.StockListing("KOSPI"), fdr.StockListing("KOSDAQ")], ignore_index=True
    )
    df = df.rename(columns={
        "Code": "티커", "Name": "종목명",
        "Close": "최신종가", "Amount": "거래대금", "Marcap": "시가총액",
    })
    for col in ("최신종가", "거래대금", "시가총액"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["티커", "최신종가", "시가총액"])
    df = df[df["시가총액"] >= MIN_MARKET_CAP]  # 잡주는 1단계에서 어차피 제외
    df = df.sort_values("거래대금", ascending=False).head(n)
    return df[["티커", "종목명", "최신종가", "시가총액"]].reset_index(drop=True)


def build_snapshot(
    date: pd.Timestamp,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    chg_pct: pd.DataFrame,
    marcap_per_close: pd.Series,
    names: dict[str, str],
) -> pd.DataFrame:
    """기준일 date의 시장 스냅샷을 일봉에서 재구성한다. (scorer 1단계 입력 형식)"""
    snap = pd.DataFrame({
        "티커":   closes.columns,
        "현재가": closes.loc[date].values,
        "등락률": chg_pct.loc[date].values,
        "거래량": volumes.loc[date].values,
    })
    snap["종목명"]   = snap["티커"].map(names)
    snap["거래대금"] = snap["현재가"] * snap["거래량"]                       # 근사
    snap["시가총액"] = snap["티커"].map(marcap_per_close) * snap["현재가"]   # 근사
    snap = snap.dropna(subset=["현재가", "등락률"])
    snap = snap[snap["현재가"] > 0]
    # 1단계 점수 동점일 때 top30 잘림이 임의로 갈리지 않도록 거래대금 내림차순 정렬
    return snap.sort_values("거래대금", ascending=False).reset_index(drop=True)


def make_bars_provider(bars: dict[str, pd.DataFrame], date: pd.Timestamp):
    """기준일까지의 일봉만 잘라 주는 provider — look-ahead 방지의 핵심."""
    lb_start = date - timedelta(days=CANDLE_LOOKBACK_DAYS)

    def provider(ticker: str) -> pd.DataFrame | None:
        df = bars.get(ticker)
        if df is None:
            return None
        sliced = df.loc[lb_start:date]
        # 기준일 당일 봉이 없으면(거래정지 등) 2단계 보강 생략 → 1단계 점수만 사용
        if len(sliced) == 0 or sliced.index[-1].normalize() != date:
            return None
        return sliced

    return provider


# ── 수익률 추적 ───────────────────────────────────────────────────

def forward_returns(
    closes_cal: pd.DataFrame,
    kospi_close: pd.Series,
    cal: pd.DatetimeIndex,
    date: pd.Timestamp,
    ticker: str,
) -> dict[str, float]:
    """기준일 종가 대비 +N거래일 종가 수익률과 같은 기간 KOSPI 수익률(%)을 반환한다."""
    i = cal.get_loc(date)
    base = closes_cal.at[date, ticker]
    out: dict[str, float] = {}
    for h in HORIZONS:
        j = i + h
        if j >= len(cal) or pd.isna(base) or base == 0:
            out[f"수익률_{h}d(%)"] = np.nan
            out[f"KOSPI_{h}d(%)"]  = np.nan
            continue
        target = cal[j]
        fwd = closes_cal.at[target, ticker]
        out[f"수익률_{h}d(%)"] = (
            round((fwd / base - 1) * 100, 2) if pd.notna(fwd) else np.nan
        )
        out[f"KOSPI_{h}d(%)"] = round(
            (kospi_close.at[target] / kospi_close.at[date] - 1) * 100, 2
        )
    return out


# ── 평가 지표 (dashboard.py에서도 import해 사용) ──────────────────

def summarize_overall(detail: pd.DataFrame) -> pd.DataFrame:
    """구간별 후보 평균 수익률 vs KOSPI, 초과수익, 승률을 집계한다."""
    rows = []
    for h in HORIZONS:
        r, k = detail[f"수익률_{h}d(%)"], detail[f"KOSPI_{h}d(%)"]
        mask = r.notna() & k.notna()
        n = int(mask.sum())
        rows.append({
            "구간": f"+{h}거래일",
            "표본": n,
            "후보 평균(%)":  round(r[mask].mean(), 2) if n else np.nan,
            "KOSPI 평균(%)": round(k[mask].mean(), 2) if n else np.nan,
            "초과수익(%p)":  round((r[mask] - k[mask]).mean(), 2) if n else np.nan,
            "승률(%)": round(float((r[mask] > k[mask]).mean()) * 100, 1) if n else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_by_score(detail: pd.DataFrame) -> pd.DataFrame:
    """점수 중앙값 기준 고득점/저득점 그룹의 구간별 평균 수익률을 비교한다."""
    med = detail["점수"].median()
    rows = []
    groups = (
        (f"고득점 (≥{med:.0f}점)", detail[detail["점수"] >= med]),
        (f"저득점 (<{med:.0f}점)", detail[detail["점수"] < med]),
    )
    for label, grp in groups:
        row: dict = {
            "그룹": label,
            "표본": len(grp),
            "평균점수": round(grp["점수"].mean(), 1) if len(grp) else np.nan,
        }
        for h in HORIZONS:
            r = grp[f"수익률_{h}d(%)"].dropna()
            row[f"+{h}d 평균(%)"] = round(r.mean(), 2) if len(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_rebound(detail: pd.DataFrame) -> pd.DataFrame:
    """'바닥 반등 후보'(2차 미분 변곡점)로 잡힌 종목만 따로 집계한다."""
    return summarize_overall(detail[detail["바닥반등"]])


# ── 실행 ─────────────────────────────────────────────────────────

def _print_table(title: str, df: pd.DataFrame) -> None:
    if df.empty or (("표본" in df.columns) and df["표본"].fillna(0).sum() == 0):
        console.print(f"[yellow]{title} — 표본 없음[/]")
        return
    table = Table(title=title, box=box.SIMPLE_HEAVY, title_justify="left")
    for col in df.columns:
        table.add_column(str(col), justify="right" if col != df.columns[0] else "left")
    for _, row in df.iterrows():
        table.add_row(*("-" if pd.isna(v) else str(v) for v in row))
    console.print(table)


def run_backtest(start: str, end: str, interval: int, universe_size: int) -> pd.DataFrame | None:
    cache = BarsCache()
    req_start = (pd.Timestamp(start) - timedelta(days=LOOKBACK_PAD)).strftime("%Y-%m-%d")
    req_end = min(
        pd.Timestamp(end) + timedelta(days=FORWARD_PAD), pd.Timestamp.today()
    ).strftime("%Y-%m-%d")

    # ── 거래일 캘린더 + 벤치마크 (KOSPI 지수) ────────────────────
    kospi = cache.get(KOSPI_INDEX, req_start, req_end)
    if kospi is None or kospi.empty:
        console.print("[bold red]KOSPI 지수(KS11) 조회 실패 — 네트워크를 확인하세요.[/]")
        return None
    cal = kospi.index.normalize()
    kospi_close = pd.Series(kospi["Close"].astype(float).values, index=cal)

    in_range = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    scoring_dates = list(in_range[::interval])
    if not scoring_dates:
        console.print(f"[bold red]{start}~{end} 사이에 거래일이 없습니다.[/]")
        return None
    console.print(
        f"기준일 {len(scoring_dates)}개: "
        + ", ".join(d.strftime("%Y-%m-%d") for d in scoring_dates)
    )

    # ── 유니버스 + 일봉 수집 ─────────────────────────────────────
    uni = load_universe(universe_size)
    console.print(f"유니버스: 현재 거래대금 상위 {len(uni)}개 종목 (시총 300억 이상)")

    bars, failed = fetch_all_bars(cache, uni["티커"].tolist(), req_start, req_end)
    if failed:
        console.print(
            f"[yellow]일봉 조회 실패 {len(failed)}종목 (건너뜀): "
            f"{', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}[/]"
        )
    if not bars:
        console.print("[bold red]일봉 데이터를 하나도 가져오지 못했습니다.[/]")
        return None

    # ── 와이드 프레임 (종가/거래량/등락률) ───────────────────────
    closes  = pd.DataFrame({t: b["Close"].astype(float)  for t, b in bars.items()}).sort_index()
    volumes = pd.DataFrame({t: b["Volume"].astype(float) for t, b in bars.items()}).sort_index()
    closes.index  = closes.index.normalize()
    volumes.index = volumes.index.normalize()
    # fill_method=None: 상장 전·거래정지로 직전 봉이 없으면 등락률 NaN → 스냅샷에서 제외
    chg_pct = closes.pct_change(fill_method=None) * 100

    last_close = closes.ffill().iloc[-1]
    marcap_now = pd.Series(uni["시가총액"].values, index=uni["티커"].values)
    marcap_per_close = (marcap_now / last_close).dropna()
    names = dict(zip(uni["티커"], uni["종목명"]))

    closes_cal = closes.reindex(cal).ffill()  # 거래정지일은 직전 체결가로 평가

    # ── 기준일별 스코어링 재현 ───────────────────────────────────
    records: list[dict] = []
    for d in scoring_dates:
        if d not in closes.index:
            console.print(f"[yellow][{d:%Y-%m-%d}] 일봉 데이터 없음 — 건너뜀[/]")
            continue
        snap = build_snapshot(d, closes, volumes, chg_pct, marcap_per_close, names)
        if snap.empty:
            console.print(f"[yellow][{d:%Y-%m-%d}] 스냅샷 비어 있음 — 건너뜀[/]")
            continue

        cands = score_candidates(snap, bars_provider=make_bars_provider(bars, d))
        if not cands:
            console.print(f"[{d:%Y-%m-%d}] 후보 없음")
            continue

        for c in cands:
            rebound = any(REBOUND_TAG in r[0] for r in c["후보이유"])
            rec = {
                "기준일":   d.strftime("%Y-%m-%d"),
                "티커":     c["티커"],
                "종목명":   c["종목명"],
                "점수":     c["점수"],
                "등락률":   c["등락률"],
                "종가":     c["현재가"],
                "거래대금(억원)": c["거래대금(억원)"],
                "바닥반등": rebound,
            }
            rec.update(forward_returns(closes_cal, kospi_close, cal, d, c["티커"]))
            records.append(rec)

        top = cands[0]
        console.print(
            f"[{d:%Y-%m-%d}] 후보 {len(cands)}개 — 1위 {top['종목명']} ({top['점수']}점)"
        )

    if not records:
        console.print("[bold red]전체 기간에서 후보가 한 건도 나오지 않았습니다.[/]")
        return None

    detail = pd.DataFrame(records)
    for h in HORIZONS:
        detail[f"초과수익_{h}d(%p)"] = (
            detail[f"수익률_{h}d(%)"] - detail[f"KOSPI_{h}d(%)"]
        ).round(2)
    return detail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="scorer.py 후보 선정 룰의 과거 성과를 검증하는 백테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", required=True, help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=5,
                        help="스코어링 재현 간격 (거래일 단위)")
    parser.add_argument("--universe", type=int, default=800,
                        help="유니버스 크기 — 현재 거래대금 상위 N개")
    args = parser.parse_args()

    if args.start > args.end:
        console.print("[bold red]--start가 --end보다 늦습니다.[/]")
        return 1
    if args.interval < 1:
        console.print("[bold red]--interval은 1 이상이어야 합니다.[/]")
        return 1

    console.rule(f"백테스트 {args.start} ~ {args.end} (간격 {args.interval}거래일)")
    detail = run_backtest(args.start, args.end, args.interval, args.universe)
    if detail is None:
        return 1

    # ── CSV 저장 ─────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"backtest_{datetime.today():%Y%m%d}.csv"
    detail.to_csv(out_path, index=False, encoding="utf-8-sig")
    console.print(f"\n상세 결과 {len(detail)}건 저장: [bold]{out_path}[/]\n")

    # ── 콘솔 요약 ────────────────────────────────────────────────
    _print_table("① 전체 후보 vs KOSPI", summarize_overall(detail))
    _print_table("② 점수 구간별 (고득점이 실제로 나았나)", summarize_by_score(detail))
    n_rebound = int(detail["바닥반등"].sum())
    _print_table(f"③ 바닥 반등 후보만 ({n_rebound}건)", summarize_rebound(detail))

    console.print(
        "[dim]※ 수익률이 아직 도래하지 않은 구간(최근 기준일의 +10/+20거래일 등)은 "
        "표본에서 제외됩니다.\n"
        "※ 유니버스는 현재 거래대금 상위 종목으로 구성되어 생존 편향이 일부 존재하며, "
        "거래대금·시가총액은 일봉 기반 근사값입니다.[/]"
    )
    return 0


if __name__ == "__main__":
    # Windows 콘솔에서 이모지·한글 깨짐 방지 (news_tracker.py와 동일 패턴)
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        console = Console(file=sys.stdout, force_terminal=True, highlight=False)
    raise SystemExit(main())
