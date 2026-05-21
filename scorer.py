"""
scorer.py — 2단계 룰 기반 후보 종목 점수·설명 생성

1단계(스냅샷 필터): market_data.load_market_data() DataFrame에서 절대 기준으로 1차 후보 30개 선정
2단계(일봉 보강):  1차 후보 30개에 한해서만 fdr.DataReader로 최근 ~60거래일 일봉을 가져와
                   거래량 배율·이동평균·RSI(14)를 계산해 점수·이유를 추가한다.

전종목 DataReader 호출을 피하는 이유:
  KOSPI+KOSDAQ 전종목(~2500개)에 DataReader를 돌리면 수십 분이 걸린다.
  1단계에서 유망 30개로 좁힌 뒤 일봉을 가져와야 현실적인 대시보드가 된다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import FinanceDataReader as fdr
    _FDR_AVAILABLE = True
except ImportError:
    _FDR_AVAILABLE = False

# ── 1단계 임계값 ─────────────────────────────────────────────────
MIN_TRADING_AMOUNT  = 5_000_000_000    # 거래대금 하한 (원) — 50억 미만 유동성 부족
HIGH_TRADING_AMOUNT = 100_000_000_000  # 거래대금 상위권 기준 (원) — 1000억 이상
MIN_MARKET_CAP      = 30_000_000_000   # 시가총액 하한 (원) — 300억 미만 잡주 제외

RATE_SAFE_MAX    = 3.0   # 안정 상승 상한 (%)
RATE_GOOD_MAX    = 7.0   # 양호 상승 상한 (%)
RATE_CAUTION_MAX = 10.0  # 주의 상승 상한 (%) — 이상이면 급등 감점

# ── 2단계 임계값 ─────────────────────────────────────────────────
CANDLE_LOOKBACK_DAYS = 90   # 조회 캘린더 일수 (≈60 거래일 보장)
VOL_MID_X            = 1.5  # 거래량 배율 — 이상이면 +15
VOL_HIGH_X           = 2.0  # 거래량 배율 — 이상이면 +20
RSI_EXTREME_OVERSOLD = 20   # 극단적 과매도 기준 — 이하이면 +20
RSI_OVERSOLD         = 30   # 과매도 기준 — 이하이면 +15
RSI_OVERBOUGHT       = 70   # 과열 기준 — 이상이면 -10

# ── 단계별 후보 수 ───────────────────────────────────────────────
STAGE1_TOP_N = 30
FINAL_TOP_N  = 10


# ── 미분 기반 지표 계산 ───────────────────────────────────────────

def _compute_derivatives(closes: pd.Series) -> dict | None:
    """5일 이동평균에 미분을 적용해 속도·가속도·모멘텀을 반환한다.

    원본 종가 대신 MA5를 먼저 구해 노이즈를 줄인 뒤 np.diff를 적용한다.
    데이터 부족(25봉 미만) 시 None 반환.

    반환 키:
      vel_avg     : 최근 5일 1차 미분 평균 (일별 상대 변화율, 소수)
      acc_current : 최신 2차 미분 값
      acc_prev    : 직전 2차 미분 값 — 부호 전환 감지에 사용
      momentum_pct: (오늘 - 20일 전) / 20일 전 × 100 (%)
    """
    if len(closes) < 25:
        return None

    # MA5로 평활화 후 NaN 제거
    ma5 = closes.rolling(5).mean().dropna().values
    if len(ma5) < 7:   # 2차 미분 최소 요건
        return None

    # 1차 미분: 일별 상대 변화율 (원 단위 차이를 이전 값으로 나눠 정규화)
    vel = np.diff(ma5) / ma5[:-1]

    # 2차 미분: 속도의 변화 방향
    acc = np.diff(vel)

    vel_avg     = float(vel[-5:].mean()) if len(vel) >= 5 else float(vel.mean())
    acc_current = float(acc[-1]) if len(acc) >= 1 else 0.0
    acc_prev    = float(acc[-2]) if len(acc) >= 2 else 0.0

    momentum_pct = float(
        (closes.iloc[-1] - closes.iloc[-21]) / closes.iloc[-21] * 100
    ) if len(closes) >= 21 else float("nan")

    return {
        "vel_avg":      vel_avg,
        "acc_current":  acc_current,
        "acc_prev":     acc_prev,
        "momentum_pct": momentum_pct,
    }


# ── RSI 계산 ─────────────────────────────────────────────────────

def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """단순 RSI(period) 계산. 데이터 부족 시 float('nan') 반환."""
    delta = closes.diff().dropna()
    if len(delta) < period:
        return float("nan")
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return float("nan")
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2)


# ── 1단계: 스냅샷 절대 기준 점수 ─────────────────────────────────

def _stage1_score(row) -> tuple[int, list[str], list[str]]:
    """장중 스냅샷 데이터만으로 점수·이유·주의점을 계산한다."""
    change_pct = float(row.get("등락률", 0) or 0)
    amount     = float(row.get("거래대금", 0) or 0)

    score:   int       = 0
    reasons: list[str] = []
    risks:   list[str] = []

    # 등락률 점수
    # +10% 이상 급등은 이미 추격매수 위험 구간이므로 감점한다.
    if 0 < change_pct <= RATE_SAFE_MAX:
        score += 15
        reasons.append(f"안정적 상승 ({change_pct:.1f}%)")
    elif RATE_SAFE_MAX < change_pct <= RATE_GOOD_MAX:
        score += 10
        reasons.append(f"양호한 상승 ({change_pct:.1f}%)")
    elif RATE_GOOD_MAX < change_pct < RATE_CAUTION_MAX:
        score += 5
        risks.append(f"강한 상승 ({change_pct:.1f}%) — 단기 과열 주의")
    elif change_pct >= RATE_CAUTION_MAX:
        score -= 15
        risks.append(f"급등 과열 ({change_pct:.1f}%) — 추격매수 위험 구간")

    # 거래대금 점수
    amount_eok = amount / 1e8
    if amount >= HIGH_TRADING_AMOUNT:
        score += 15
        reasons.append(f"시장 상위권 거래대금 ({amount_eok:,.0f}억원)")
    elif amount >= MIN_TRADING_AMOUNT:
        score += 10
        reasons.append(f"충분한 거래대금 ({amount_eok:,.0f}억원)")

    return score, reasons, risks


# ── 2단계: 일봉 기반 보강 점수 ───────────────────────────────────

def _stage2_enrich(ticker: str) -> tuple[int, list[str], list[str]]:
    """fdr.DataReader로 일봉을 가져와 추가 점수·이유·주의점을 반환한다.

    네트워크 오류나 데이터 부족 등 어떤 이유로든 실패하면 (0, [], [])을 반환한다.
    호출자는 실패해도 1단계 점수만으로 그대로 처리하면 된다.
    """
    if not _FDR_AVAILABLE:
        return 0, [], []

    try:
        start = (datetime.today() - timedelta(days=CANDLE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        df = fdr.DataReader(ticker, start)

        # 20일 이동평균 + RSI(14) 계산에 최소 21개 필요
        if df is None or len(df) < 21:
            return 0, [], []

        closes  = df["Close"].astype(float)
        volumes = df["Volume"].astype(float)

        score:   int       = 0
        reasons: list[str] = []
        risks:   list[str] = []

        # ── 거래량 배율 (오늘 ÷ 직전 20일 평균) ──────────────────
        today_vol  = volumes.iloc[-1]
        avg_vol_20 = volumes.iloc[-21:-1].mean()   # 오늘 제외 직전 20일

        if avg_vol_20 > 0:
            vol_ratio = today_vol / avg_vol_20
            if vol_ratio >= VOL_HIGH_X:
                score += 20
                reasons.append(
                    f"거래량 급증 ({vol_ratio:.1f}배 — 20일 평균 {avg_vol_20:,.0f}주 대비)"
                )
            elif vol_ratio >= VOL_MID_X:
                score += 15
                reasons.append(
                    f"거래량 증가 ({vol_ratio:.1f}배 — 20일 평균 {avg_vol_20:,.0f}주 대비)"
                )

        # ── 20일 이동평균선 위/아래 ───────────────────────────────
        ma20          = closes.iloc[-21:-1].mean()
        current_close = closes.iloc[-1]

        if current_close > ma20:
            score += 10
            reasons.append(f"20일 이동평균선 상회 (현재 {current_close:,.0f}원 > MA20 {ma20:,.0f}원)")
        else:
            risks.append(f"20일 이동평균선 하회 (현재 {current_close:,.0f}원 < MA20 {ma20:,.0f}원)")

        # ── RSI(14) ───────────────────────────────────────────────
        rsi = _compute_rsi(closes)

        if not pd.isna(rsi):
            if rsi <= RSI_EXTREME_OVERSOLD:
                score += 20
                reasons.append(f"RSI {rsi:.1f} — 극단적 과매도, 강한 반등 가능성")
            elif rsi <= RSI_OVERSOLD:
                score += 15
                reasons.append(f"RSI {rsi:.1f} — 과매도 구간, 반등 가능성")
            elif rsi >= RSI_OVERBOUGHT:
                score -= 10
                risks.append(f"RSI {rsi:.1f} — 과열 구간, 단기 조정 주의")
            else:
                reasons.append(f"RSI {rsi:.1f} — 중립 구간")

        # ── 미분 기반 지표 (속도·가속도·모멘텀) ──────────────────
        deriv = _compute_derivatives(closes)
        if deriv is not None:
            vel_avg      = deriv["vel_avg"]
            acc_current  = deriv["acc_current"]
            acc_prev     = deriv["acc_prev"]
            momentum_pct = deriv["momentum_pct"]

            vel_pct = vel_avg * 100   # 일별 평균 변화율(%)로 표시용 변환

            # 가속도 부호 전환 여부 (핵심 신호)
            inflection = acc_current > 0 and acc_prev <= 0

            # 반등 신호 조건: 가속도 양전환 + 직전 속도가 -1.0%/일 이하 (진짜 하락)
            # vel_avg < 0이지만 -0.01보다 크면(예: -0.12%/일) 횡보로 간주, 반등 불인정
            REAL_DOWNTREND_THRESHOLD = -0.01   # -1.0%/일

            if inflection and vel_avg <= REAL_DOWNTREND_THRESHOLD:
                # 뚜렷하게 하락하던 중 가속도 음→양 전환 → 바닥 변곡점
                score += 20
                reasons.append(
                    f"하락세가 꺾이고 반등 신호 — 바닥 변곡점 가능성"
                    f" (속도 {vel_pct:+.2f}%/일, 가속도 양전환)"
                )
            elif inflection and vel_avg < 0:
                # 가속도 전환은 있지만 하락폭이 미미 → 횡보 변곡, 점수 없음
                reasons.append(
                    f"횡보 중 방향 전환 신호 (속도 {vel_pct:+.2f}%/일 — 반등으로 보기엔 하락폭 부족)"
                )
            elif inflection and vel_avg >= 0:
                # 상승 중 가속도도 재전환 → 상승 재가속
                score += 10
                reasons.append(
                    f"상승 재가속 신호 — 모멘텀 회복"
                    f" (속도 {vel_pct:+.2f}%/일)"
                )
            elif acc_current > 0 and vel_avg > 0:
                # 상승 + 가속 지속
                score += 10
                reasons.append(f"상승 가속 중 (속도 {vel_pct:+.2f}%/일)")
            elif acc_current < 0 and vel_avg > 0:
                # 상승 중이지만 둔화
                risks.append(
                    f"상승 모멘텀 약화 중 (속도 {vel_pct:+.2f}%/일, 둔화)"
                )
            elif acc_current < 0 and vel_avg < 0:
                # 하락 가속 중 — 이중 경고
                score -= 10
                risks.append(
                    f"하락 가속 중 — 추가 하락 경계 (속도 {vel_pct:+.2f}%/일)"
                )

            # 20일 모멘텀
            if not pd.isna(momentum_pct):
                if momentum_pct >= 10:
                    score += 15
                    reasons.append(f"20일 모멘텀 강세 ({momentum_pct:+.1f}%)")
                elif momentum_pct >= 3:
                    score += 10
                    reasons.append(f"20일 모멘텀 양호 ({momentum_pct:+.1f}%)")
                elif momentum_pct <= -10:
                    score -= 10
                    risks.append(f"20일 모멘텀 약세 ({momentum_pct:+.1f}%)")
                else:
                    reasons.append(f"20일 모멘텀 중립 ({momentum_pct:+.1f}%)")

        return score, reasons, risks

    except Exception:
        return 0, [], []


# ── 공개 인터페이스 ───────────────────────────────────────────────

def score_candidates(df: pd.DataFrame) -> list[dict]:
    """2단계 룰로 후보 종목을 선정하고 최종 상위 10개를 반환한다.

    Parameters
    ----------
    df : market_data.load_market_data()가 반환한 DataFrame
         필요 컬럼: 티커, 종목명, 현재가, 등락률, 거래대금, 시가총액, 거래량

    Returns
    -------
    list[dict] — 합산 점수 내림차순 상위 10개
        keys: 티커, 종목명, 현재가, 등락률, 거래대금(억원), 점수, 후보이유, 주의점
    """

    # ── 1단계: 스냅샷 필터 → top 30 ──────────────────────────────
    filtered = df.copy()
    if "거래대금" in filtered.columns:
        filtered = filtered[filtered["거래대금"] >= MIN_TRADING_AMOUNT]
    if "시가총액" in filtered.columns:
        filtered = filtered[filtered["시가총액"] >= MIN_MARKET_CAP]

    stage1: list[dict] = []
    for _, row in filtered.iterrows():
        s1, reasons, risks = _stage1_score(row)
        if s1 <= 0:
            continue

        amount = float(row.get("거래대금", 0) or 0)
        stage1.append({
            "티커":          str(row.get("티커", "")),
            "종목명":        str(row.get("종목명", "")),
            "현재가":        int(float(row.get("현재가", 0) or 0)),
            "등락률":        round(float(row.get("등락률", 0) or 0), 2),
            "거래대금(억원)": int(amount / 1e8),
            "점수":          s1,
            "후보이유":      reasons,
            "주의점":        risks,
        })

    stage1.sort(key=lambda x: x["점수"], reverse=True)
    top30 = stage1[:STAGE1_TOP_N]

    # ── 2단계: 일봉 보강 (top 30에만 적용) ───────────────────────
    for c in top30:
        s2, s2_reasons, s2_risks = _stage2_enrich(c["티커"])
        c["점수"]    += s2
        c["후보이유"] += s2_reasons
        c["주의점"]   += s2_risks

    top30.sort(key=lambda x: x["점수"], reverse=True)
    return top30[:FINAL_TOP_N]
