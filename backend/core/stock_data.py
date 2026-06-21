"""
backend/core/stock_data.py

주가 데이터 수집 및 날짜별 감성 점수 집계 유틸리티 모듈.
FinanceDataReader(FDR)를 사용해 최근 7일간의 일별 종가(Close)를 가져오고,
뉴스 아이템 리스트로부터 날짜별 평균 감성 점수를 계산합니다.

※ yfinance 대신 FinanceDataReader를 사용하는 이유:
    - 네이버 금융 기반으로 국내 통신 환경에서 빠르고 안정적인 응답
    - Yahoo Finance의 Rate Limit · 타임아웃 문제 없음
    - 6자리 종목 코드만으로 조회 가능 (접미사 불필요)
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import FinanceDataReader as fdr
import pandas as pd

logger = logging.getLogger(__name__)

# ── 종목명 → FinanceDataReader 종목 코드 매핑 테이블 ─────────────────────────────
# FDR은 한국 주식에 순수 6자리 숫자 코드를 사용합니다. (예: "005930")
TICKER_MAP: Dict[str, str] = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "LG에너지솔루션": "373220",
    "삼성바이오로직스": "207940",
    "현대차": "005380",
    "현대자동차": "005380",
    "기아": "000270",
    "기아차": "000270",
    "POSCO홀딩스": "005490",
    "포스코홀딩스": "005490",
    "LG화학": "051910",
    "NAVER": "035420",
    "네이버": "035420",
    "카카오": "035720",
    "삼성SDI": "006400",
    "현대모비스": "012330",
    "KB금융": "105560",
    "신한지주": "055550",
    "하나금융지주": "086790",
    "우리금융지주": "316140",
    "LG전자": "066570",
    "삼성물산": "028260",
    "SK텔레콤": "017670",
    "KT": "030200",
    "LG유플러스": "032640",
    "셀트리온": "068270",
    "카카오뱅크": "323410",
    "크래프톤": "259960",
    "두산에너빌리티": "034020",
    "한국전력": "015760",
}

# 뉴스 감성 레이블 → 수치화 스코어 매핑
# KoELECTRA는 0.0~1.0 범주의 확률값을 반환합니다.
# ≥ 0.6: 긍정(+1), ≤ 0.4: 부정(-1), 나머지: 중립(0)
SENTIMENT_POSITIVE_THRESHOLD = 0.6
SENTIMENT_NEGATIVE_THRESHOLD = 0.4
SENTIMENT_SCORE_POSITIVE = 1.0
SENTIMENT_SCORE_NEUTRAL = 0.0
SENTIMENT_SCORE_NEGATIVE = -1.0


def resolve_ticker(stock_name: str) -> Optional[str]:
    """
    종목명을 FinanceDataReader 종목 코드로 변환합니다.

    TICKER_MAP에서 직접 매핑을 시도하고, 없으면 None을 반환합니다.

    Args:
        stock_name: 한국어 종목명 (예: "삼성전자")

    Returns:
        6자리 종목 코드 문자열 (예: "005930"), 매핑 실패 시 None
    """
    ticker = TICKER_MAP.get(stock_name.strip())
    if not ticker:
        logger.warning("Ticker not found in TICKER_MAP for stock_name='%s'", stock_name)
    return ticker


def _quantify_sentiment(raw_score: float) -> float:
    """
    KoELECTRA 원시 감성 확률값(0.0~1.0)을 삼분류 수치로 변환합니다.

    Args:
        raw_score: KoELECTRA가 반환한 감성 확률 (0.0 = 부정, 1.0 = 긍정)

    Returns:
        수치화된 감성 점수: 1.0(긍정) / 0.0(중립) / -1.0(부정)
    """
    if raw_score >= SENTIMENT_POSITIVE_THRESHOLD:
        return SENTIMENT_SCORE_POSITIVE
    elif raw_score <= SENTIMENT_NEGATIVE_THRESHOLD:
        return SENTIMENT_SCORE_NEGATIVE
    return SENTIMENT_SCORE_NEUTRAL


def _parse_date_from_string(date_str: str) -> Optional[str]:
    """
    다양한 날짜 문자열 포맷에서 'YYYY-MM-DD' 형식의 날짜를 파싱합니다.

    지원 포맷:
        - ISO 8601: "2026-06-20T10:30:00+09:00"
        - 날짜만:   "2026-06-20"
        - 네이버 스타일: "2026.06.20"

    Args:
        date_str: 파싱 대상 날짜 문자열

    Returns:
        'YYYY-MM-DD' 포맷 날짜 문자열, 파싱 실패 시 None
    """
    if not date_str:
        return None

    # ISO 8601 포맷: 'YYYY-MM-DDTHH:MM:SS...' 또는 'YYYY-MM-DD HH:MM:SS'
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        return f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"

    # 네이버 뉴스 스타일: 'YYYY.MM.DD'
    dot_match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", date_str)
    if dot_match:
        return f"{dot_match.group(1)}-{dot_match.group(2)}-{dot_match.group(3)}"

    logger.debug("Could not parse date from string: '%s'", date_str)
    return None


def compute_daily_sentiment(
    news_list: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    뉴스 아이템 리스트로부터 날짜별 평균 감성 점수를 계산합니다.

    각 뉴스의 'date' 필드를 파싱하고, 'sentiment_score'를 삼분류 수치(-1/0/+1)로
    변환한 뒤 날짜별로 평균을 산출합니다.

    Args:
        news_list: 뉴스 딕셔너리 리스트.
                   각 딕셔너리는 'date'(str)와 'sentiment_score'(float) 키를 포함해야 합니다.

    Returns:
        날짜 문자열('YYYY-MM-DD')을 키로, 평균 감성 점수를 값으로 하는 딕셔너리.
        예: {"2026-06-20": 0.5, "2026-06-19": -0.33}
    """
    daily_scores: Dict[str, List[float]] = defaultdict(list)

    for news in news_list:
        raw_date = news.get("date", "")
        raw_score = news.get("sentiment_score")

        parsed_date = _parse_date_from_string(raw_date)
        if not parsed_date or raw_score is None:
            continue

        quantified = _quantify_sentiment(float(raw_score))
        daily_scores[parsed_date].append(quantified)

    # 날짜별 평균 계산
    return {
        day: round(sum(scores) / len(scores), 4)
        for day, scores in daily_scores.items()
        if scores
    }


def fetch_stock_prices(ticker: str, period_days: int = 10) -> Dict[str, float]:
    """
    FinanceDataReader를 사용하여 최근 N일간의 일별 종가(Close)를 조회합니다.

    네이버 금융 기반 FDR을 사용하므로 Yahoo Finance의 Rate Limit 및
    타임아웃 문제 없이 안정적으로 국내 주가 데이터를 수집합니다.

    거래소 공휴일 및 주말을 감안하여 실제 7 거래일을 확보하기 위해
    period_days + 5 일 전부터 조회를 시작합니다.

    Args:
        ticker: FinanceDataReader 종목 코드 (예: "005930")
        period_days: 조회할 최대 달력 일수 (기본값: 10)

    Returns:
        날짜 문자열('YYYY-MM-DD')을 키로, 종가(float)를 값으로 하는 딕셔너리.
        조회 실패 시 빈 딕셔너리를 반환합니다.

    Raises:
        ValueError: 티커가 비어 있는 경우
    """
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")

    try:
        # 주말·공휴일 보정을 위해 period_days + 5일 전부터 조회
        start_date = (
            datetime.now() - timedelta(days=period_days + 5)
        ).strftime("%Y-%m-%d")

        # FinanceDataReader 호출 (네이버 금융 기반, 6자리 종목 코드 사용)
        df: pd.DataFrame = fdr.DataReader(ticker, start=start_date)

        if df.empty:
            logger.warning(
                "FinanceDataReader returned empty DataFrame for ticker='%s'", ticker
            )
            return {}

        # 인덱스를 'YYYY-MM-DD' 문자열로 변환
        prices: Dict[str, float] = {}
        for ts, row in df.iterrows():
            date_str = ts.strftime("%Y-%m-%d")
            prices[date_str] = round(float(row["Close"]), 2)

        logger.info(
            "Fetched %d day(s) of price data from FDR for ticker='%s'",
            len(prices),
            ticker,
        )
        return prices

    except Exception as exc:
        logger.error(
            "Failed to fetch stock prices from FDR for ticker='%s': %s",
            ticker,
            exc,
            exc_info=True,
        )
        return {}


def build_time_series(
    stock_name: str,
    news_list: List[Dict[str, Any]],
    lookback_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    주가 데이터와 뉴스 감성 데이터를 날짜 기준으로 병합하여 시계열 리스트를 생성합니다.

    최근 lookback_days 거래일 내의 날짜를 대상으로 주가와 감성 점수를 조인합니다.
    주가 데이터가 있는 날짜를 기준으로 생성하며, 해당 날짜에 감성 데이터가 없으면
    sentiment_score는 None으로 채웁니다.

    Args:
        stock_name: 한국어 종목명 (예: "삼성전자")
        news_list: FastAPI /analyze 응답의 news_list (date, sentiment_score 포함)
        lookback_days: 최근 몇 거래일을 표시할지 지정 (기본값: 7)

    Returns:
        시계열 포인트 딕셔너리의 리스트 (날짜 오름차순 정렬).
        각 딕셔너리 포맷:
            {
                "date": "YYYY-MM-DD",
                "price": float | None,
                "sentiment_score": float | None,
            }
        종목 코드 미지원 시 빈 리스트를 반환합니다.
    """
    ticker = resolve_ticker(stock_name)
    if not ticker:
        logger.warning(
            "Cannot build time series: no ticker found for stock_name='%s'", stock_name
        )
        return []

    # FDR에서 최근 lookback_days + 여유(3일) 데이터 로드
    prices = fetch_stock_prices(ticker, period_days=lookback_days + 3)
    if not prices:
        logger.warning("No price data available for ticker='%s'", ticker)
        return []

    # 최근 lookback_days 거래일만 필터링 (날짜 내림차순 → 상위 N개 → 다시 오름차순)
    sorted_dates = sorted(prices.keys(), reverse=True)
    recent_dates = sorted(sorted_dates[:lookback_days])

    # 날짜별 감성 평균 계산
    daily_sentiment = compute_daily_sentiment(news_list)

    # 주가 기준 날짜로 조인
    time_series: List[Dict[str, Any]] = []
    for day in recent_dates:
        time_series.append(
            {
                "date": day,
                "price": prices.get(day),
                "sentiment_score": daily_sentiment.get(day),
            }
        )

    return time_series
