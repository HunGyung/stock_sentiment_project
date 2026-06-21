import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import torch

# Add project root directory to Python path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.scraper import scrape_news
from backend.core.stock_data import build_time_series
from backend.core.rag_service import StockRAGService
from backend.models.sentiment import KoELECTRASentimentAnalyzer
from backend.models.summarizer import KoBARTSummarizer
from backend.models.gemini_client import generate_report_async

# ── 로거 설정 ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── 전역 모델 인스턴스 싱글톤 관리 ──────────────────────────────────────────────────
# FastAPI lifespan 이벤트를 사용하여 애플리케이션 시작 시 1회만 모델을 로드하여 메모리에 상주시키고,
# 모든 API 요청 시 전역 인스턴스를 공유하도록 설계합니다.
models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션 수명 주기 관리 (Startup & Shutdown)"""
    logger.info("Initializing models on startup (Singleton Load)...")
    try:
        # 감성 분석 모델 (KoELECTRA) 로드
        models["sentiment"] = KoELECTRASentimentAnalyzer()
        # 문서 요약 모델 (KoBART) 로드
        models["summarizer"] = KoBARTSummarizer()
        logger.info("All AI models loaded successfully and stored in memory.")
    except Exception as exc:
        logger.critical(f"Failed to initialize models on startup: {exc}", exc_info=True)
        raise exc

    # ── RAG 서비스 싱글톤 초기화 ─────────────────────────────────────────────────
    # ChromaDB / sentence-transformers 미설치 환경에서도 서버가 기동될 수 있도록
    # 비치명적(Non-critical) 예외로 처리합니다. RAG 실패 시 None으로 폴백됩니다.
    try:
        models["rag"] = StockRAGService()
        logger.info("ChromaDB RAG Service initialized successfully.")
    except Exception as rag_err:
        logger.error(
            "Failed to load RAG Service (Non-critical, will continue without RAG): %s",
            rag_err,
        )
        models["rag"] = None
    # ────────────────────────────────────────────────────────────────────────────

    yield

    # 서버 종료 시 리소스 정리
    logger.info("Shutting down server. Clearing model resources...")
    models.clear()


app = FastAPI(
    title="Stock Sentiment Assistant API",
    description="FastAPI backend for crawling financial news, generating summaries, analyzing sentiments, and writing AI reports.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS 미들웨어 적용 ──────────────────────────────────────────────────────────
# Streamlit 등 다른 오리진의 프론트엔드 연동을 지원하기 위해 CORS 허용 설정을 추가합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API 엔드포인트 설계 ──────────────────────────────────────────────────────────

@app.get("/api/analyze")
async def analyze_stock(
    stock_name: str = Query(..., description="분석 대상 주식 종목명 (예: 삼성전자)"),
    limit: int = Query(15, description="크롤링할 뉴스 기사의 개수 제한 (기본값: 15)"),
):
    """
    주어진 주식 종목에 대한 뉴스를 비동기적으로 수집 및 분석합니다.
    
    전체 파이프라인:
      1. scrape_news 비동기 뉴스 검색 및 원문 크롤링
      2. KoBART 기반 3줄 요약 수행
      3. KoELECTRA 기반 감성 스코어 분석 (기사 제목 기준)
      4. 감성 스코어 통계 산출
      5. 비동기 generate_report_async를 통한 Gemini AI 분석 보고서 생성
    """
    logger.info("Analysis request received | Stock: %s | Limit: %d", stock_name, limit)
    
    if not stock_name.strip():
        raise HTTPException(
            status_code=400,
            detail="stock_name Query parameter cannot be empty or whitespace.",
        )
        
    try:
        # 1. 비동기 뉴스 스크랩 수행
        logger.info("Running Step 1: News scraping...")
        raw_news = await scrape_news(stock_name, limit)
        
        if not raw_news:
            logger.warning("No news articles found for stock: %s", stock_name)
            return {
                "stock_name": stock_name,
                "avg_sentiment_score": 0.5,
                "total_news_count": 0,
                "news_list": [],
                "report": "## [알림] 리포트 생성 불가\n\n해당 종목명에 대해 최근 수집된 뉴스가 없습니다. 종목명을 확인해 주세요.",
            }

        # 전역 싱글톤 모델 인스턴스 확인
        sentiment_analyzer: KoELECTRASentimentAnalyzer = models.get("sentiment")
        summarizer: KoBARTSummarizer = models.get("summarizer")
        
        if not sentiment_analyzer or not summarizer:
            raise RuntimeError("AI model resources are not loaded or initialized.")

        # 2. 개별 뉴스 요약 및 감성 분석 수행
        logger.info("Running Step 2: Summarization & Sentiment analysis (KoBART & KoELECTRA)...")
        news_list: List[Dict[str, Any]] = []
        sentiment_scores: List[float] = []

        for item in raw_news:
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()
            
            # 본문 요약 생성
            summary = summarizer.summarize(body)
            # 제목에 대한 감성 지수 분석 (제목이 텍스트의 핵심 감성을 가장 신속하게 투영함)
            sentiment_score = sentiment_analyzer.analyze(title)
            sentiment_scores.append(sentiment_score)
            
            news_list.append({
                "title": title,
                "link": item.get("link", ""),
                "date": item.get("date", ""),
                "publisher": item.get("publisher", ""),
                "summary": summary,
                "sentiment_score": round(sentiment_score, 4),
            })

        # 평균 감성 지수 연산
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5

        # ── Step 2.5: ChromaDB RAG 인덱싱 ───────────────────────────────────────
        # news_list에 body(원문)가 없는 경우 raw_news에서 보완하여 인덱싱합니다.
        rag_service: StockRAGService | None = models.get("rag")
        if rag_service is not None:
            try:
                # raw_news의 body를 news_list 항목에 병합 (RAG 인덱싱에 원문 본문 활용)
                news_for_rag = [
                    {**news, "body": raw.get("body", "")}
                    for news, raw in zip(news_list, raw_news)
                ]
                loop = asyncio.get_event_loop()
                indexed_count = await loop.run_in_executor(
                    None,
                    lambda: rag_service.index_news(news_for_rag, stock_name),
                )
                logger.info(
                    "Running Step 2.5: Indexed %d chunks into ChromaDB for stock='%s'",
                    indexed_count, stock_name,
                )
            except Exception as rag_err:
                logger.error(
                    "Failed to index news into ChromaDB (Non-critical): %s", rag_err
                )
        # ────────────────────────────────────────────────────────────────────────

        # 3. Gemini API 기반 종합 리포트 생성 (RAG 인스턴스 주입)
        logger.info("Running Step 3: Generating final AI investment report via Gemini API (RAG: %s)...",
                    "ON" if rag_service is not None else "OFF")
        report_markdown = await generate_report_async(
            stock_name,
            news_list,
            rag_service=rag_service,
        )

        # 4. 주가-감성 시계열 데이터 생성
        logger.info("Running Step 4: Building price-sentiment time series...")
        time_series = build_time_series(stock_name, news_list, lookback_days=7)

        return {
            "stock_name": stock_name,
            "avg_sentiment_score": round(avg_sentiment_score, 4),
            "total_news_count": len(news_list),
            "news_list": news_list,
            "report": report_markdown,
            "time_series": time_series,
        }

    except Exception as exc:
        logger.error("Error occurred in the stock sentiment analysis backend pipeline: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error in analysis pipeline: {type(exc).__name__}: {exc}",
        )


@app.get("/api/timeseries")
async def get_timeseries(
    stock_name: str = Query(..., description="분석 대상 주식 종목명 (예: 삼성전자)"),
    limit: int = Query(15, description="크롤링할 뉴스 기사의 개수 제한 (기본값: 15)"),
    lookback_days: int = Query(7, description="조회할 최근 거래일 수 (기본값: 7)"),
):
    """
    주어진 종목에 대해 최근 N 거래일의 주가(종가)와 일별 뉴스 평균 감성 점수를
    시계열 포맷으로 반환합니다.

    뉴스 크롤링 → KoELECTRA 감성 분석 → 날짜별 평균 감성 집계 → yfinance 주가 병합
    의 순서로 처리됩니다.

    Args:
        stock_name: 분석할 종목명
        limit: 수집할 뉴스 기사 수 상한
        lookback_days: 조회할 거래일 수 (기본값: 7)

    Returns:
        JSON 응답:
        {
            "stock_name": str,
            "ticker": str | None,
            "time_series": [
                {"date": "YYYY-MM-DD", "price": float | None, "sentiment_score": float | None},
                ...
            ]
        }
    """
    logger.info(
        "Timeseries request received | Stock: %s | Limit: %d | Lookback: %d days",
        stock_name, limit, lookback_days,
    )

    if not stock_name.strip():
        raise HTTPException(
            status_code=400,
            detail="stock_name Query parameter cannot be empty or whitespace.",
        )

    try:
        # 뉴스 크롤링 수행
        raw_news = await scrape_news(stock_name.strip(), limit)

        sentiment_analyzer: KoELECTRASentimentAnalyzer = models.get("sentiment")
        if not sentiment_analyzer:
            raise RuntimeError("Sentiment model is not loaded.")

        # 제목 기반 감성 분석만 수행 (요약·리포트 생략 → 응답 속도 향상)
        news_list: List[Dict[str, Any]] = []
        for item in raw_news:
            title = item.get("title", "").strip()
            if not title:
                continue
            sentiment_score = sentiment_analyzer.analyze(title)
            news_list.append({
                "title": title,
                "date": item.get("date", ""),
                "sentiment_score": round(sentiment_score, 4),
            })

        # 주가-감성 시계열 생성
        time_series = build_time_series(stock_name.strip(), news_list, lookback_days=lookback_days)

        from backend.core.stock_data import resolve_ticker
        return {
            "stock_name": stock_name.strip(),
            "ticker": resolve_ticker(stock_name.strip()),
            "time_series": time_series,
        }

    except Exception as exc:
        logger.error(
            "Error occurred in timeseries endpoint for stock='%s': %s",
            stock_name, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error in timeseries pipeline: {type(exc).__name__}: {exc}",
        )


@app.get("/api/health")
async def health_check():
    """
    서버 및 로드된 AI 모델들의 헬스 상태를 반환합니다.
    """
    sentiment_ok = "sentiment" in models and models["sentiment"] is not None
    summarizer_ok = "summarizer" in models and models["summarizer"] is not None
    rag_ok = models.get("rag") is not None

    is_healthy = sentiment_ok and summarizer_ok
    status = "healthy" if is_healthy else "unhealthy"

    return {
        "status": status,
        "details": {
            "sentiment_analyzer_loaded": sentiment_ok,
            "summarizer_loaded": summarizer_ok,
            "rag_service_loaded": rag_ok,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
    }
