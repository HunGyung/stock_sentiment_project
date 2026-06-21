"""
backend/models/gemini_client.py
================================
Gemini API 기반 주식 투자 리포트 생성 클라이언트.

환경변수 GEMINI_API_KEY가 설정되어 있어야 합니다.
FastAPI의 비동기 엔드포인트와 직접 연동 가능하도록 async 인터페이스를 제공합니다.

사용 예:
    import asyncio
    from backend.models.gemini_client import generate_report_async

    report = asyncio.run(generate_report_async("삼성전자", news_list))
    print(report)
"""

import asyncio
import logging
import os
import sys
from typing import Any

# 프로젝트 루트를 Python path에 추가 (스크립트 단독 실행 지원)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# google-genai SDK (google-generativeai 후속 공식 패키지)
from google import genai
from google.genai import types

# ── 로거 설정 ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── 상수 ───────────────────────────────────────────────────────────────────────
_DEFAULT_MODEL = "gemini-1.5-flash"
_SENTIMENT_POSITIVE_THRESHOLD = 0.6
_SENTIMENT_NEGATIVE_THRESHOLD = 0.4
_ERROR_FALLBACK_TEMPLATE = (
    "## [ERROR] 리포트 생성 실패\n\n"
    "Gemini API 호출 중 오류가 발생했습니다.\n\n"
    "**원인**: {reason}\n\n"
    "잠시 후 다시 시도하거나, API 키 및 네트워크 상태를 확인해 주세요."
)


# ── 내부 헬퍼 함수 ────────────────────────────────────────────────────────────
def _load_api_key() -> str:
    """
    환경변수 GEMINI_API_KEY에서 API 키를 로드합니다.

    Returns:
        API 키 문자열

    Raises:
        ValueError: 환경변수가 설정되지 않았거나 빈 값인 경우
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "환경변수 'GEMINI_API_KEY'가 설정되지 않았습니다. "
            "Gemini API 키를 발급받아 환경변수로 등록한 뒤 다시 실행해 주세요.\n"
            "  Windows PowerShell : $env:GEMINI_API_KEY='your-key-here'\n"
            "  Linux/macOS        : export GEMINI_API_KEY='your-key-here'"
        )
    return api_key


def _classify_sentiment(avg_score: float) -> str:
    """
    평균 감성 점수를 긍정/중립/부정 문자열로 변환합니다.

    Args:
        avg_score: 0.0(매우 부정) ~ 1.0(매우 긍정) 사이의 평균 감성 점수

    Returns:
        "긍정적", "중립적", "부정적" 중 하나
    """
    if avg_score >= _SENTIMENT_POSITIVE_THRESHOLD:
        return "긍정적"
    if avg_score <= _SENTIMENT_NEGATIVE_THRESHOLD:
        return "부정적"
    return "중립적"


def _build_prompt(
    stock_name: str,
    news_list: list[dict[str, Any]],
    retrieved_positives: list[str] | None = None,
    retrieved_negatives: list[str] | None = None,
) -> str:
    """
    Gemini에 전송할 프롬프트를 구성합니다.

    RAG로 검색된 호재·악재 문맥이 제공될 경우 프롬프트 내 전용 섹션에 동적으로 삽입하여
    Gemini가 구체적인 근거에 기반한 보고서를 생성하도록 유도합니다.

    Args:
        stock_name: 분석 대상 종목명 (예: "삼성전자")
        news_list:  뉴스 딕셔너리 리스트.
                    각 항목은 title, summary, sentiment_score, publisher, date 키를 포함.
        retrieved_positives: RAG 유사도 검색으로 추출한 호재 관련 뉴스 텍스트 조각 리스트.
                             None 또는 빈 리스트이면 대체 문구를 삽입합니다.
        retrieved_negatives: RAG 유사도 검색으로 추출한 악재 관련 뉴스 텍스트 조각 리스트.
                             None 또는 빈 리스트이면 대체 문구를 삽입합니다.

    Returns:
        완성된 프롬프트 문자열

    Raises:
        ValueError: news_list가 비어 있는 경우
    """
    if not news_list:
        raise ValueError("news_list가 비어 있습니다. 최소 1개 이상의 뉴스를 제공해 주세요.")

    # 감성 점수 통계 계산
    scores = [float(item.get("sentiment_score", 0.5)) for item in news_list]
    avg_score      = sum(scores) / len(scores)
    positive_count = sum(1 for s in scores if s >= _SENTIMENT_POSITIVE_THRESHOLD)
    negative_count = sum(1 for s in scores if s <= _SENTIMENT_NEGATIVE_THRESHOLD)
    neutral_count  = len(scores) - positive_count - negative_count
    sentiment_label = _classify_sentiment(avg_score)

    # 뉴스 항목 포맷팅
    news_blocks: list[str] = []
    for i, item in enumerate(news_list, start=1):
        score       = float(item.get("sentiment_score", 0.5))
        score_label = _classify_sentiment(score)
        block = (
            f"[뉴스 {i}]\n"
            f"  제목     : {item.get('title', '(제목 없음)')}\n"
            f"  요약     : {item.get('summary', '(요약 없음)')}\n"
            f"  감성점수 : {score:.3f} ({score_label})\n"
            f"  언론사   : {item.get('publisher', '미상')} | 날짜: {item.get('date', '미상')}"
        )
        news_blocks.append(block)

    news_section = "\n\n".join(news_blocks)

    # ── RAG 문맥 섹션 빌드 ──────────────────────────────────────────────────────
    if retrieved_positives:
        pos_context_section = "\n".join([f"- {ctx.strip()}" for ctx in retrieved_positives])
    else:
        pos_context_section = "- 수집된 뉴스에서 뚜렷한 상승 호재 맥락이 검색되지 않았습니다."

    if retrieved_negatives:
        neg_context_section = "\n".join([f"- {ctx.strip()}" for ctx in retrieved_negatives])
    else:
        neg_context_section = "- 수집된 뉴스에서 뚜렷한 하락 리스크 맥락이 검색되지 않았습니다."
    # ────────────────────────────────────────────────────────────────────────────

    prompt = f"""당신은 대한민국 주식 시장 전문 애널리스트입니다.
아래 제공된 뉴스 데이터 및 RAG 의미 검색 상세 문맥을 바탕으로, 객관적이고 깊이 있는 투자 분석 보고서를 마크다운으로 작성해 주세요.

=== 분석 대상 ===
종목명       : {stock_name}
수집 뉴스 수  : {len(news_list)}건
평균 감성 점수 : {avg_score:.3f} / 1.000  ({sentiment_label})
  - 긍정 뉴스: {positive_count}건 / 중립 뉴스: {neutral_count}건 / 부정 뉴스: {negative_count}건

=== [핵심 근거] RAG 호재/상승 요인 문맥 ===
{pos_context_section}

=== [핵심 근거] RAG 악재/리스크 요인 문맥 ===
{neg_context_section}

=== 뉴스 요약 목록 ===
{news_section}

=== 보고서 작성 지침 ===
반드시 아래 마크다운 구조를 정확히 따르세요. 각 섹션은 빠짐없이 작성해야 합니다.
핵심 위주로 가독성 있게 서술하고, 불필요한 미사여구와 반복 표현은 배제해 주세요.
위의 'RAG 호재/상승 요인 문맥'과 'RAG 악재/리스크 요인 문맥'을 Momentum 및 Risk 섹션의
핵심 근거로 반드시 활용하여 구체적으로 서술하세요.

# [{stock_name}] AI 기반 뉴스 감성 & 투자 분석 보고서

## [감성 분석 종합]
- 수집된 뉴스의 평균 감성 점수와 긍/부정 성향을 분석하여 서술하세요. (3문장 이내 요약)
- 긍정/중립/부정 뉴스 비율을 언급하고 시장 분위기를 간결하게 마무리하세요.

## [핵심 호재 - Momentum]
- RAG 호재 문맥과 긍정 뉴스를 근거로, 상승 요인을 핵심 위주로 정리하세요.
- 글머리 기호(-)로 3~5개 항목, 항목별 최대 2문장으로 작성하세요.
- 근거가 없으면 "분석된 호재 없음"으로 명시하세요.

## [핵심 악재 - Risk]
- RAG 악재 문맥과 부정 뉴스를 근거로, 하락 리스크를 핵심 위주로 정리하세요.
- 글머리 기호(-)로 3~5개 항목, 항목별 최대 2문장으로 작성하세요.
- 근거가 없으면 "분석된 악재 없음"으로 명시하세요.

## [종합 투자 가이드]
- **단기 관점**: 1~4주 내 투자 전략을 간결하게 제시하세요. (2문장 이내)
- **중장기 관점**: 1~6개월 투자 전략을 간결하게 제시하세요. (2문장 이내)
- **주의사항**: 본 보고서는 AI 기술을 활용한 참고 자료일 뿐이며, 모든 투자 판단에 대한 최종 결정과 책임은 투자자 본인에게 있다는 면책 조항 문장을 정확히 작성하세요.

위 구조와 지침을 엄수하여 보고서를 한국어로 작성해 주세요.
"""
    return prompt


# ── 공개 비동기 인터페이스 ────────────────────────────────────────────────────
async def generate_report_async(
    stock_name: str,
    news_list: list[dict[str, Any]],
    rag_service: Any = None,
    model_name: str = _DEFAULT_MODEL,
) -> str:
    """
    Gemini API를 이용하여 주식 투자 분석 보고서를 비동기적으로 생성합니다.

    rag_service가 제공되면 보고서 생성 전 호재/악재 쿼리로 ChromaDB에서
    의미 기반 Top-3 문맥을 검색하고, _build_prompt에 동적으로 주입합니다.

    Args:
        stock_name: 분석 대상 종목명 (예: "삼성전자")
        news_list:  뉴스 정보 딕셔너리 리스트. 각 항목은 다음 키를 포함해야 합니다:
                    - title (str)            : 기사 제목
                    - summary (str)          : KoBART 요약문
                    - sentiment_score (float): 감성 스코어 (0.0 ~ 1.0)
                    - publisher (str)         : 언론사명
                    - date (str)             : 날짜 문자열
        rag_service: StockRAGService 인스턴스 (선택). None이면 RAG 없이 기존 방식으로 생성.
        model_name: 사용할 Gemini 모델명 (기본값: "gemini-1.5-flash")

    Returns:
        마크다운 형식의 투자 분석 보고서 문자열.
        오류 발생 시 대체 에러 안내 문자열을 반환합니다.

    Raises:
        ValueError: GEMINI_API_KEY 환경변수 미설정 시 (호출 전에 즉시 발생)
    """
    # API 키 로드 — 누락 시 즉시 ValueError 발생 (상위로 전파)
    api_key = _load_api_key()

    try:
        # gemini-1.5-flash → gemini-2.5-flash 자동 매핑 (v1beta 지원 중단 대응)
        if model_name == "gemini-1.5-flash":
            logger.info("모델 'gemini-1.5-flash'는 지원하지 않으므로 'gemini-2.5-flash'로 자동 매핑하여 진행합니다.")
            model_name = "gemini-2.5-flash"

        logger.info(
            "Gemini 보고서 생성 시작 | 종목: %s | 모델: %s | 뉴스 수: %d건 | RAG: %s",
            stock_name, model_name, len(news_list),
            "ON" if rag_service is not None else "OFF",
        )

        # ── RAG 의미 검색 실행 ─────────────────────────────────────────────────
        # ChromaDB의 동기 API를 run_in_executor로 이벤트 루프 블로킹 없이 실행합니다.
        # 호재/악재 두 쿼리를 병렬로 실행하여 지연을 최소화합니다.
        retrieved_pos: list[str] = []
        retrieved_neg: list[str] = []

        if rag_service is not None:
            try:
                loop = asyncio.get_event_loop()

                # 호재·악재 쿼리를 동시에 병렬 실행
                pos_future = loop.run_in_executor(
                    None,
                    lambda: rag_service.retrieve_contexts(
                        "최근 실적 개선, HBM 공급 확대, 기술력 확보 등 핵심 호재 및 긍정적 모멘텀 요인",
                        stock_name,
                        top_k=3,
                    ),
                )
                neg_future = loop.run_in_executor(
                    None,
                    lambda: rag_service.retrieve_contexts(
                        "대외 경제 불안, 원가 상승, 적자 지속, 소송 또는 규제, 파업 등 핵심 악재 및 리스크 요인",
                        stock_name,
                        top_k=3,
                    ),
                )

                retrieved_pos, retrieved_neg = await asyncio.gather(
                    pos_future, neg_future, return_exceptions=True
                )

                # gather 반환값이 예외 객체일 경우 빈 리스트로 대체
                if isinstance(retrieved_pos, Exception):
                    logger.error("RAG 호재 검색 실패: %s", retrieved_pos)
                    retrieved_pos = []
                if isinstance(retrieved_neg, Exception):
                    logger.error("RAG 악재 검색 실패: %s", retrieved_neg)
                    retrieved_neg = []

                logger.info(
                    "RAG 검색 완료 | 호재 %d건, 악재 %d건",
                    len(retrieved_pos), len(retrieved_neg),
                )

            except Exception as rag_exc:
                logger.error(
                    "RAG 컨텍스트 검색 중 예외 발생 (보고서 생성 계속): %s",
                    rag_exc, exc_info=True,
                )
        # ────────────────────────────────────────────────────────────────────────

        # google-genai SDK 클라이언트 초기화
        client = genai.Client(api_key=api_key)

        # RAG 문맥을 포함한 프롬프트 구성
        prompt = _build_prompt(
            stock_name,
            news_list,
            retrieved_positives=retrieved_pos or None,
            retrieved_negatives=retrieved_neg or None,
        )

        # 동기 API 호출을 비동기 루프에서 실행 (이벤트 루프 블로킹 방지)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=4096,
                ),
            ),
        )

        report = response.text.strip()
        logger.info(
            "Gemini 보고서 생성 완료 | 종목: %s | 길이: %d자",
            stock_name, len(report),
        )
        return report

    except ValueError:
        # API 키 누락 / news_list 오류는 상위로 재발생
        raise

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Gemini API 호출 실패 | 종목: %s | 원인: %s",
            stock_name, reason,
        )
        return _ERROR_FALLBACK_TEMPLATE.format(reason=reason)


# ── 스크립트 단독 실행 테스트 ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows 터미널 UTF-8 출력 설정
    sys.stdout.reconfigure(encoding="utf-8")

    # 더미 뉴스 데이터 (실제 KoBART 요약 + 감성 점수 시뮬레이션)
    dummy_news: list[dict[str, Any]] = [
        {
            "title": "삼성전자, 2분기 영업이익 10조 원 돌파...시장 예상치 상회",
            "summary": (
                "삼성전자가 2분기 영업이익 10조 원을 돌파하며 시장 예상치를 크게 웃돌았다. "
                "반도체 부문 회복과 HBM 수요 급증이 주요 원인으로 꼽힌다."
            ),
            "sentiment_score": 0.87,
            "publisher": "한국경제",
            "date": "2024-07-05",
        },
        {
            "title": "삼성전자 HBM3E, 엔비디아 품질 테스트 최종 승인 임박",
            "summary": (
                "삼성전자의 HBM3E 메모리가 엔비디아의 최종 품질 인증을 앞두고 있어 "
                "AI 반도체 시장 점유율 확대가 기대된다."
            ),
            "sentiment_score": 0.82,
            "publisher": "매일경제",
            "date": "2024-07-04",
        },
        {
            "title": "삼성전자, 파운드리 사업 적자 지속...2나노 수율 개선 난항",
            "summary": (
                "삼성전자 파운드리 부문이 2나노 공정 수율 문제로 적자가 지속되고 있으며 "
                "TSMC와의 격차 해소에 시간이 걸릴 것으로 분석된다."
            ),
            "sentiment_score": 0.24,
            "publisher": "서울경제",
            "date": "2024-07-03",
        },
        {
            "title": "외국인 투자자, 삼성전자 순매수 전환...코스피 견인",
            "summary": (
                "외국인 투자자들이 삼성전자를 중심으로 순매수 기조로 전환하며 "
                "코스피 지수 상승을 이끌었다."
            ),
            "sentiment_score": 0.71,
            "publisher": "연합뉴스",
            "date": "2024-07-02",
        },
        {
            "title": "삼성전자 노조, 창사 이래 첫 파업 예고...생산 차질 우려",
            "summary": (
                "삼성전자 최대 노동조합이 임금 협상 결렬로 창사 이래 첫 파업을 예고했다. "
                "반도체 생산 라인 일부 차질이 우려된다."
            ),
            "sentiment_score": 0.18,
            "publisher": "조선비즈",
            "date": "2024-07-01",
        },
    ]

    SEP = "=" * 60
    print(SEP)
    print("Gemini API 투자 리포트 생성 테스트")
    print(SEP)
    print(f"대상 종목  : 삼성전자")
    print(f"뉴스 수    : {len(dummy_news)}건")
    print(f"사용 모델  : {_DEFAULT_MODEL}")
    print(SEP)
    print()

    try:
        report = asyncio.run(generate_report_async("삼성전자", dummy_news))
        print(report)
        print()
        print(SEP)
        print("테스트 완료!")
    except ValueError as ve:
        print(f"[설정 오류] {ve}")
    except Exception as e:
        print(f"[예상치 못한 오류] {type(e).__name__}: {e}")
