"""
market_data.py — FinanceDataReader 기반 시장 데이터 로딩
"""
from datetime import datetime, timedelta
import hashlib
import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

try:
    import FinanceDataReader as fdr
    _FDR_AVAILABLE = True
except ImportError:
    _FDR_AVAILABLE = False


def get_latest_trading_date() -> str | None:
    """대표 지수 일봉에서 실제 최신 거래일(YYYY-MM-DD)을 반환한다."""
    if not _FDR_AVAILABLE:
        return None
    try:
        start = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        index_bars = fdr.DataReader("KS11", start)
        if index_bars is not None and not index_bars.empty:
            return pd.Timestamp(index_bars.index[-1]).strftime("%Y-%m-%d")
    except Exception as exc:
        logger.warning("최신 거래일 조회 실패: %s", exc)
    return None


def market_snapshot_key(df: pd.DataFrame, trading_date: str | None) -> str:
    """후보 캐시 무효화를 위한 시장 스냅샷 지문을 반환한다."""
    columns = [
        column for column in ("티커", "현재가", "등락률", "거래대금", "시가총액")
        if column in df.columns
    ]
    if not columns or df.empty:
        return f"{trading_date or ''}:empty"
    values = pd.util.hash_pandas_object(df[columns], index=False).values.tobytes()
    digest = hashlib.sha256(values).hexdigest()[:16]
    return f"{trading_date or ''}:{len(df)}:{digest}"


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data():
    """KOSPI + KOSDAQ 전종목 시장 데이터를 로드한다. (캐시 5분)

    Returns:
        (df, date_str) — df는 통합 DataFrame, date_str은 기준일 문자열.
        오류 시 (None, None).
    """
    if not _FDR_AVAILABLE:
        return None, None

    try:
        kospi = fdr.StockListing("KOSPI")
        kosdaq = fdr.StockListing("KOSDAQ")

        df = pd.concat([kospi, kosdaq], ignore_index=True)

        # 필요한 컬럼만 정리
        df = df.rename(columns={
            "Code": "티커",
            "Name": "종목명",
            "Close": "현재가",
            "ChagesRatio": "등락률",
            "Changes": "전일대비",
            "Amount": "거래대금",
            "Marcap": "시가총액",
            "Volume": "거래량",
        })

        cols = ["티커", "종목명", "현재가", "등락률", "전일대비", "거래대금", "시가총액", "거래량"]
        df = df[[c for c in cols if c in df.columns]].copy()

        # 숫자형 강제 변환 (결측치 제거)
        for col in ["현재가", "등락률", "거래대금", "시가총액"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["현재가", "등락률"])
        df = df[df["현재가"] > 0]

        date_str = get_latest_trading_date()
        df.attrs["market_date"] = date_str
        df.attrs["snapshot_at"] = datetime.now().isoformat(timespec="seconds")
        return df, date_str

    except Exception as e:
        logger.exception("시장 데이터 로딩 실패: %s", e)
        return None, None
