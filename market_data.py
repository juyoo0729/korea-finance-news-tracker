"""
market_data.py — FinanceDataReader 기반 시장 데이터 로딩
"""

from datetime import datetime

import pandas as pd
import streamlit as st

try:
    import FinanceDataReader as fdr
    _FDR_AVAILABLE = True
except ImportError:
    _FDR_AVAILABLE = False


def get_latest_trading_date() -> str:
    """오늘 날짜를 기준으로 기준일 문자열(YYYY-MM-DD) 반환.

    FinanceDataReader는 항상 가장 최근 거래일 데이터를 반환하므로
    날짜는 로드 후 실제 데이터에서 읽는다.
    """
    return datetime.now().strftime("%Y-%m-%d")


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

        date_str = datetime.now().strftime("%Y-%m-%d")
        return df, date_str

    except Exception as e:
        return None, None
