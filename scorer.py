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
import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")

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
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2)


# ── 1단계: 스냅샷 절대 기준 점수 ─────────────────────────────────

def _stage1_score(row) -> tuple[int, list[tuple[str, int]], list[tuple[str, int]]]:
    """장중 스냅샷 데이터만으로 점수·이유·주의점을 계산한다."""
    change_pct = float(row.get("등락률", 0) or 0)
    amount     = float(row.get("거래대금", 0) or 0)

    score:   int                    = 0
    reasons: list[tuple[str, int]]  = []
    risks:   list[tuple[str, int]]  = []

    # 등락률 점수
    # +10% 이상 급등은 이미 단기 과열 구간이므로 감점한다.
    if 0 < change_pct <= RATE_SAFE_MAX:
        score += 15
        reasons.append((f"안정적 상승 ({change_pct:.1f}%)", 15))
    elif RATE_SAFE_MAX < change_pct <= RATE_GOOD_MAX:
        score += 10
        reasons.append((f"양호한 상승 ({change_pct:.1f}%)", 10))
    elif RATE_GOOD_MAX < change_pct < RATE_CAUTION_MAX:
        score += 5
        risks.append((f"강한 상승 ({change_pct:.1f}%) — 단기 과열 주의", 5))
    elif change_pct >= RATE_CAUTION_MAX:
        score -= 15
        risks.append((f"급등 과열 ({change_pct:.1f}%) — 단기 변동성 확대 구간", -15))
    elif change_pct <= -10:
        score -= 25
        risks.append((f"당일 급락 ({change_pct:.1f}%) — 변동성·추가 하락 위험", -25))
    elif change_pct <= -5:
        score -= 15
        risks.append((f"당일 큰 폭 하락 ({change_pct:.1f}%) — 반전 확인 필요", -15))
    elif change_pct < 0:
        risks.append((f"당일 하락 ({change_pct:.1f}%)", 0))

    # 거래대금 점수
    amount_eok = amount / 1e8
    if amount >= HIGH_TRADING_AMOUNT:
        score += 15
        reasons.append((f"시장 상위권 거래대금 ({amount_eok:,.0f}억원)", 15))
    elif amount >= MIN_TRADING_AMOUNT:
        score += 10
        reasons.append((f"충분한 거래대금 ({amount_eok:,.0f}억원)", 10))

    return score, reasons, risks


# ── 2단계: 일봉 기반 보강 점수 ───────────────────────────────────

def _default_bars(ticker: str) -> pd.DataFrame | None:
    """오늘 기준 최근 CANDLE_LOOKBACK_DAYS 캘린더 일수의 일봉을 fdr로 조회한다."""
    if not _FDR_AVAILABLE:
        return None
    start = (datetime.today() - timedelta(days=CANDLE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return fdr.DataReader(ticker, start)


def _stage2_enrich(
    ticker: str,
    bars_provider=None,
) -> tuple[int, list[tuple[str, int]], list[tuple[str, int]], dict]:
    """일봉을 가져와 추가 점수·이유·주의점을 반환한다.

    bars_provider: 티커 → 일봉 DataFrame(Close/Volume 컬럼, 마지막 행이 "기준일")을
                   반환하는 콜러블. None이면 오늘 기준 fdr 조회(기존 동작).
                   backtest.py가 과거 시점 재현(look-ahead 방지)에 사용한다.

    점수·이유·주의점과 함께 완료 상태·일봉 기준일·오류 정보를 반환한다.
    """
    meta = {
        "2단계완료": False,
        "2단계상태": "데이터 없음",
        "2단계오류": "",
        "일봉기준일": "",
        "수집일봉기준일": "",
        "미완성봉가능": False,
    }
    try:
        df = (bars_provider or _default_bars)(ticker)

        if df is None:
            return 0, [], [], meta
        if len(df) < 21:
            meta["2단계상태"] = "데이터 부족"
            meta["2단계오류"] = f"일봉 {len(df)}개 (최소 21개 필요)"
            return 0, [], [], meta
        missing = {"Close", "Volume"} - set(df.columns)
        if missing:
            raise KeyError(f"필수 일봉 컬럼 누락: {', '.join(sorted(missing))}")

        try:
            latest_bar = pd.Timestamp(df.index[-1])
            meta["수집일봉기준일"] = latest_bar.strftime("%Y-%m-%d")
            now = datetime.now(KST)
            meta["미완성봉가능"] = bool(
                latest_bar.date() == now.date()
                and now.weekday() < 5
                and (now.hour, now.minute) < (15, 40)
            )
        except (TypeError, ValueError):
            meta["수집일봉기준일"] = str(df.index[-1])

        analysis_df = df.iloc[:-1] if meta["미완성봉가능"] else df
        if len(analysis_df) < 21:
            meta["2단계상태"] = "데이터 부족"
            meta["2단계오류"] = f"완성 일봉 {len(analysis_df)}개 (최소 21개 필요)"
            return 0, [], [], meta
        meta["일봉기준일"] = pd.Timestamp(analysis_df.index[-1]).strftime("%Y-%m-%d")

        closes  = analysis_df["Close"].astype(float)
        volumes = analysis_df["Volume"].astype(float)

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
                reasons.append((
                    f"거래량 급증 ({vol_ratio:.1f}배 — 20일 평균 {avg_vol_20:,.0f}주 대비)", 20
                ))
            elif vol_ratio >= VOL_MID_X:
                score += 15
                reasons.append((
                    f"거래량 증가 ({vol_ratio:.1f}배 — 20일 평균 {avg_vol_20:,.0f}주 대비)", 15
                ))
        if meta["미완성봉가능"]:
            reasons.append(("장중 미완성 최신봉 제외 후 완성봉으로 분석", 0))

        # ── 20일 이동평균선 위/아래 ───────────────────────────────
        ma20          = closes.iloc[-21:-1].mean()
        current_close = closes.iloc[-1]

        if current_close > ma20:
            score += 10
            reasons.append((f"20일 이동평균선 상회 (현재 {current_close:,.0f}원 > MA20 {ma20:,.0f}원)", 10))
        else:
            risks.append((f"20일 이동평균선 하회 (현재 {current_close:,.0f}원 < MA20 {ma20:,.0f}원)", 0))

        # ── RSI(14) ───────────────────────────────────────────────
        rsi = _compute_rsi(closes)

        if not pd.isna(rsi):
            if rsi <= RSI_EXTREME_OVERSOLD:
                score += 20
                reasons.append((f"RSI {rsi:.1f} — 극단적 과매도 구간으로 분류됨", 20))
            elif rsi <= RSI_OVERSOLD:
                score += 15
                reasons.append((f"RSI {rsi:.1f} — 과매도 구간으로 분류됨", 15))
            elif rsi >= RSI_OVERBOUGHT:
                score -= 10
                risks.append((f"RSI {rsi:.1f} — 과열 구간으로 분류됨 (주의 구간)", -10))
            else:
                reasons.append((f"RSI {rsi:.1f} — 중립 구간", 0))

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
                reasons.append((
                    f"하락 추세 중 가속도 양전환 (바닥 변곡점 관찰 조건)"
                    f" (속도 {vel_pct:+.2f}%/일)", 20
                ))
            elif inflection and vel_avg < 0:
                # 가속도 전환은 있지만 하락폭이 미미 → 횡보 변곡, 점수 없음
                reasons.append((
                    f"횡보 중 방향 전환 (속도 {vel_pct:+.2f}%/일 — 변곡 판정 기준 미달)", 0
                ))
            elif inflection and vel_avg >= 0:
                # 상승 중 가속도도 재전환 → 상승 재가속
                score += 10
                reasons.append((
                    f"상승 재가속 상태 — 모멘텀 회복 (속도 {vel_pct:+.2f}%/일)", 10
                ))
            elif acc_current > 0 and vel_avg > 0:
                # 상승 + 가속 지속
                score += 10
                reasons.append((f"상승 가속 상태 (속도 {vel_pct:+.2f}%/일)", 10))
            elif acc_current < 0 and vel_avg > 0:
                # 상승 중이지만 둔화
                risks.append((f"상승 모멘텀 둔화 상태 (속도 {vel_pct:+.2f}%/일)", 0))
            elif acc_current < 0 and vel_avg < 0:
                # 하락 가속 중 — 이중 경고
                score -= 10
                risks.append((f"하락 가속 상태 — 단기 변동성 확대 구간 (속도 {vel_pct:+.2f}%/일)", -10))

            # 20일 모멘텀
            if not pd.isna(momentum_pct):
                if momentum_pct >= 10:
                    score += 15
                    reasons.append((f"20일 모멘텀 강세 ({momentum_pct:+.1f}%)", 15))
                elif momentum_pct >= 3:
                    score += 10
                    reasons.append((f"20일 모멘텀 양호 ({momentum_pct:+.1f}%)", 10))
                elif momentum_pct <= -10:
                    score -= 10
                    risks.append((f"20일 모멘텀 약세 ({momentum_pct:+.1f}%)", -10))
                else:
                    reasons.append((f"20일 모멘텀 중립 ({momentum_pct:+.1f}%)", 0))

        meta["2단계완료"] = True
        meta["2단계상태"] = "완료"
        return score, reasons, risks, meta

    except Exception as exc:
        logger.warning("일봉 분석 실패 ticker=%s: %s", ticker, exc)
        meta["2단계상태"] = "조회 실패"
        meta["2단계오류"] = f"{type(exc).__name__}: {exc}"
        return 0, [], [], meta


# ── 공개 인터페이스 ───────────────────────────────────────────────

def score_candidates(
    df: pd.DataFrame,
    bars_provider=None,
    snapshot_date: str | None = None,
) -> list[dict]:
    """2단계 룰로 후보 종목을 선정하고 최종 상위 10개를 반환한다.

    Parameters
    ----------
    df : market_data.load_market_data()가 반환한 DataFrame
         필요 컬럼: 티커, 종목명, 현재가, 등락률, 거래대금, 시가총액, 거래량
    bars_provider : 티커 → 일봉 DataFrame을 반환하는 콜러블 (선택).
         None이면 오늘 기준 fdr 조회. backtest.py가 과거 시점 재현에 사용.
    snapshot_date : 1단계 시장 스냅샷의 기준 거래일(선택).

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
        market_cap = float(row.get("시가총액", 0) or 0)
        stage1.append({
            "티커":          str(row.get("티커", "")),
            "종목명":        str(row.get("종목명", "")),
            "현재가":        int(float(row.get("현재가", 0) or 0)),
            "등락률":        round(float(row.get("등락률", 0) or 0), 2),
            "거래대금(억원)": int(amount / 1e8),
            "점수":          s1,
            "후보이유":      reasons,
            "주의점":        risks,
            "_거래대금원":   amount,
            "_시가총액원":   market_cap,
        })

    stage1.sort(key=lambda x: (
        -x["점수"], -x["_거래대금원"], -x["_시가총액원"], x["티커"]
    ))
    top30 = stage1[:STAGE1_TOP_N]

    # ── 2단계: 일봉 보강 (top 30에만 적용) ───────────────────────
    for c in top30:
        s2, s2_reasons, s2_risks, meta = _stage2_enrich(c["티커"], bars_provider)
        c["점수"]    += s2
        c["후보이유"] += s2_reasons
        c["주의점"]   += s2_risks
        c.update(meta)
        c["스냅샷기준일"] = snapshot_date or ""
        c["기준일불일치"] = bool(
            snapshot_date
            and c["일봉기준일"]
            and snapshot_date != c["일봉기준일"]
        )
        reason_text = " ".join(text for text, _ in c["후보이유"])
        if "과매도" in reason_text or "하락 추세 중 가속도 양전환" in reason_text:
            c["관찰유형"] = "과매도 반전 관찰"
        elif "거래량 급증" in reason_text or "거래량 증가" in reason_text:
            c["관찰유형"] = "거래량 이상 관찰"
        elif any(keyword in reason_text for keyword in ("상승", "모멘텀", "이동평균선 상회")):
            c["관찰유형"] = "추세 지속 관찰"
        else:
            c["관찰유형"] = "조건 충족 관찰"

    top30.sort(key=lambda x: (
        -x["점수"], not x["2단계완료"], -x["_거래대금원"],
        -x["_시가총액원"], x["티커"],
    ))
    result = top30[:FINAL_TOP_N]
    completed_count = sum(candidate["2단계완료"] for candidate in top30)
    for candidate in result:
        candidate["2단계완료수"] = completed_count
        candidate["2단계대상수"] = len(top30)
        candidate.pop("_거래대금원", None)
        candidate.pop("_시가총액원", None)
    return result
