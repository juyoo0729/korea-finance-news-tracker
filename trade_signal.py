"""
trade_signal.py — 종목의 단기 변동성·기술지표 상태 관찰

scorer.py와 동일한 지표(RSI 14, MA5/MA20, 미분 기반 변곡점)를 사용해
일봉 차트 위에 과거의 조건 발생 지점을 표시하고, 오늘 기준 시장 상태와
관찰 기준선(하단 기준선·상단 기준선·MA20)을 계산한다.

이 모듈은 매수·매도 추천이나 매매 타이밍 지시를 하지 않는다.
지표 조건이 언제 발생했는지 분류·시각화하는 것이 전부이며,
해석과 판단은 사용자 몫이다.

조건 분류 룰:
  상방 전환 이벤트 — MA5가 MA20 상향 돌파, RSI 30 상향 통과(과매도 구간 이탈),
                     하락 추세 중 가속도 양전환(바닥 변곡점 조건)
  하방 전환 이벤트 — MA5가 MA20 하향 이탈, RSI 70 하향 통과(과열 구간 이탈)
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
    """차트에 표시할 과거 조건 발생 이벤트를 날짜순으로 반환한다.

    반환 항목: {"date", "price", "side"("상방"|"하방"), "label"}
    """
    events: list[dict] = []
    ma5, ma20, rsi, close = df["MA5"], df["MA20"], df["RSI"], df["Close"]

    for i in range(1, len(df)):
        if pd.isna(ma20.iloc[i]) or pd.isna(ma20.iloc[i - 1]):
            continue
        date, price = df.index[i], float(close.iloc[i])

        # 이동평균 교차
        if ma5.iloc[i - 1] <= ma20.iloc[i - 1] and ma5.iloc[i] > ma20.iloc[i]:
            events.append({"date": date, "price": price, "side": "상방",
                           "label": "MA5가 MA20 상향 돌파"})
        elif ma5.iloc[i - 1] >= ma20.iloc[i - 1] and ma5.iloc[i] < ma20.iloc[i]:
            events.append({"date": date, "price": price, "side": "하방",
                           "label": "MA5가 MA20 하향 이탈"})

        # RSI 기준값 통과
        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
            continue
        if rsi.iloc[i - 1] <= RSI_OVERSOLD < rsi.iloc[i]:
            events.append({"date": date, "price": price, "side": "상방",
                           "label": f"RSI 30 상향 통과 ({rsi.iloc[i]:.0f})"})
        elif rsi.iloc[i - 1] >= RSI_OVERBOUGHT > rsi.iloc[i]:
            events.append({"date": date, "price": price, "side": "하방",
                           "label": f"RSI 70 하향 통과 ({rsi.iloc[i]:.0f})"})

    return events


def _current_market_state(df: pd.DataFrame) -> tuple[str, list[str]]:
    """오늘 기준 시장 상태('상방 조건 우세'|'하방 리스크 우세'|'중립')와 근거를 반환한다."""
    close = float(df["Close"].iloc[-1])
    ma20 = float(df["MA20"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else None

    up_pts: list[str] = []
    down_pts: list[str] = []

    if rsi is not None:
        if rsi <= RSI_OVERSOLD:
            up_pts.append(f"RSI {rsi:.0f} — 과매도 구간으로 분류됨")
        elif rsi >= RSI_OVERBOUGHT:
            down_pts.append(f"RSI {rsi:.0f} — 과열 구간으로 분류됨")

    if close > ma20:
        up_pts.append(f"종가가 MA20({ma20:,.0f}원) 위 — 단기 추세 기준선 상회")
    else:
        down_pts.append(f"종가가 MA20({ma20:,.0f}원) 아래 — 단기 추세 기준선 하회")

    deriv = _compute_derivatives(df["Close"])
    if deriv is not None:
        inflection = deriv["acc_current"] > 0 and deriv["acc_prev"] <= 0
        if inflection and deriv["vel_avg"] <= -0.01:
            up_pts.append("하락 추세 중 가속도 양전환 — 바닥 변곡점 조건 발생")
        elif deriv["acc_current"] < 0 and deriv["vel_avg"] < 0:
            down_pts.append("하락 가속 상태 — 단기 변동성 확대 구간으로 분류됨")

    # 최근 3거래일 내 이동평균 교차 이벤트도 현재 상태 분류에 반영
    recent = [e for e in _detect_events(df.iloc[-4:].copy()) if "MA" in e["label"]]
    for e in recent:
        (up_pts if e["side"] == "상방" else down_pts).append(f"최근 {e['label']} 발생")

    if len(up_pts) > len(down_pts):
        state = "상방 조건 우세"
    elif len(down_pts) > len(up_pts):
        state = "하방 리스크 우세"
    else:
        state = "중립"
    return state, up_pts + down_pts


def analyze_market_state(ticker: str, bars_provider=None) -> dict | None:
    """티커의 단기 변동성·지표 상태 분석 결과를 반환한다. 데이터 부족·오류 시 None.

    반환 키:
      bars   : Close/MA5/MA20/RSI 컬럼이 추가된 일봉 DataFrame
      events : 차트 마커용 과거 조건 발생 이벤트 목록 (상방/하방)
      state  : "상방 조건 우세" | "하방 리스크 우세" | "중립"
      reasons: 상태 분류 근거 문자열 목록 (관찰형 서술)
      levels : {"하단 기준선(20일 저가)", "상단 기준선(60일 고가)", "MA20"} — 관찰 기준선
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

        state, reasons = _current_market_state(df)
        lows = df["Low"].astype(float) if "Low" in df.columns else df["Close"]
        highs = df["High"].astype(float) if "High" in df.columns else df["Close"]

        levels = {
            "하단 기준선(20일 저가)": float(lows.iloc[-20:].min()),
            "상단 기준선(60일 고가)": float(highs.iloc[-60:].max()),
            "MA20": float(df["MA20"].iloc[-1]),
        }

        return {
            "bars": df,
            "events": _detect_events(df),
            "state": state,
            "reasons": reasons,
            "levels": levels,
        }
    except Exception:
        return None
