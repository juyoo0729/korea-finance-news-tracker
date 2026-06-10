"""
trade_signal.py — 후보 종목의 매수·매도 타이밍 분석

scorer.py와 동일한 지표(RSI 14, MA5/MA20, 미분 기반 변곡점)를 사용해
일봉 차트 위에 과거 매수·매도 신호를 표시하고, 오늘 기준 종합 신호와
매매 참고가(지지선·저항선·MA20)를 계산한다.

신호 룰:
  매수 — 골든크로스(MA5가 MA20 상향 돌파), RSI 30 상향 탈출(과매도 해소),
         하락세 중 가속도 양전환(바닥 변곡점)
  매도 — 데드크로스(MA5가 MA20 하향 돌파), RSI 70 하향 이탈(과열 해소)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scorer import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    _compute_derivatives,
    _default_bars,
)


def _rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta).clip(lower=0).rolling(period).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(losses.notna(), np.nan)


def _detect_events(df: pd.DataFrame) -> list[dict]:
    """차트에 표시할 과거 매수·매도 이벤트를 날짜순으로 반환한다.

    반환 항목: {"date", "price", "side"("매수"|"매도"), "label"}
    """
    events: list[dict] = []
    ma5, ma20, rsi, close = df["MA5"], df["MA20"], df["RSI"], df["Close"]

    for i in range(1, len(df)):
        if pd.isna(ma20.iloc[i]) or pd.isna(ma20.iloc[i - 1]):
            continue
        date, price = df.index[i], float(close.iloc[i])

        # 이동평균 교차
        if ma5.iloc[i - 1] <= ma20.iloc[i - 1] and ma5.iloc[i] > ma20.iloc[i]:
            events.append({"date": date, "price": price, "side": "매수",
                           "label": "골든크로스 (MA5 > MA20)"})
        elif ma5.iloc[i - 1] >= ma20.iloc[i - 1] and ma5.iloc[i] < ma20.iloc[i]:
            events.append({"date": date, "price": price, "side": "매도",
                           "label": "데드크로스 (MA5 < MA20)"})

        # RSI 구간 탈출
        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
            continue
        if rsi.iloc[i - 1] <= RSI_OVERSOLD < rsi.iloc[i]:
            events.append({"date": date, "price": price, "side": "매수",
                           "label": f"RSI 과매도 탈출 ({rsi.iloc[i]:.0f})"})
        elif rsi.iloc[i - 1] >= RSI_OVERBOUGHT > rsi.iloc[i]:
            events.append({"date": date, "price": price, "side": "매도",
                           "label": f"RSI 과열 이탈 ({rsi.iloc[i]:.0f})"})

    return events


def _current_signal(df: pd.DataFrame) -> tuple[str, list[str]]:
    """오늘 기준 종합 신호('매수 우위'|'매도 우위'|'관망')와 근거를 반환한다."""
    close = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else None

    buy_pts: list[str] = []
    sell_pts: list[str] = []

    if rsi is not None:
        if rsi <= RSI_OVERSOLD:
            buy_pts.append(f"RSI {rsi:.0f} — 과매도 구간, 분할 매수 고려")
        elif rsi >= RSI_OVERBOUGHT:
            sell_pts.append(f"RSI {rsi:.0f} — 과열 구간, 보유 시 차익실현 고려")

    if close > ma20:
        buy_pts.append(f"종가가 MA20({ma20:,.0f}원) 위 — 상승 추세 유지")
    else:
        sell_pts.append(f"종가가 MA20({ma20:,.0f}원) 아래 — 추세 이탈 주의")

    deriv = _compute_derivatives(df["Close"])
    if deriv is not None:
        inflection = deriv["acc_current"] > 0 and deriv["acc_prev"] <= 0
        if inflection and deriv["vel_avg"] <= -0.01:
            buy_pts.append("하락세 중 가속도 양전환 — 바닥 변곡점 가능성")
        elif deriv["acc_current"] < 0 and deriv["vel_avg"] < 0:
            sell_pts.append("하락 가속 중 — 신규 진입 비추천")

    # 최근 3거래일 내 교차 이벤트도 현재 신호에 반영
    recent = [e for e in _detect_events(df.iloc[-4:].copy()) if "크로스" in e["label"]]
    for e in recent:
        (buy_pts if e["side"] == "매수" else sell_pts).append(f"최근 {e['label']} 발생")

    if len(buy_pts) > len(sell_pts):
        signal = "매수 우위"
    elif len(sell_pts) > len(buy_pts):
        signal = "매도 우위"
    else:
        signal = "관망"
    return signal, buy_pts + sell_pts


def analyze_trade_timing(ticker: str, bars_provider=None) -> dict | None:
    """티커의 매매 타이밍 분석 결과를 반환한다. 데이터 부족·오류 시 None.

    반환 키:
      bars   : Close/MA5/MA20/RSI 컬럼이 추가된 일봉 DataFrame
      events : 차트 마커용 과거 매수·매도 이벤트 목록
      signal : "매수 우위" | "매도 우위" | "관망"
      reasons: 신호 근거 문자열 목록
      levels : {"지지선(20일 저가)", "저항선(60일 고가)", "MA20"} — 매매 참고가
    """
    try:
        df = (bars_provider or _default_bars)(ticker)
        if df is None or len(df) < 25:
            return None

        df = df.copy()
        df["Close"] = df["Close"].astype(float)
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["RSI"] = _rsi_series(df["Close"])

        signal, reasons = _current_signal(df)
        lows = df["Low"].astype(float) if "Low" in df.columns else df["Close"]
        highs = df["High"].astype(float) if "High" in df.columns else df["Close"]

        levels = {
            "지지선(20일 저가)": float(lows.iloc[-20:].min()),
            "저항선(60일 고가)": float(highs.iloc[-60:].max()),
            "MA20": float(df["MA20"].iloc[-1]),
        }

        return {
            "bars": df,
            "events": _detect_events(df),
            "signal": signal,
            "reasons": reasons,
            "levels": levels,
        }
    except Exception:
        return None
