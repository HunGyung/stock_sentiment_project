import sys
import os
from typing import Any, Dict, List
import streamlit as st
import httpx

# Add project root to python path for modular import compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 상수 설정 ──────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/api/analyze"
HEALTH_CHECK_URL = "http://127.0.0.1:8000/api/health"

# ── UI 스타일링 헬퍼 ───────────────────────────────────────────────────────────
def get_sentiment_badge(score: float) -> str:
    """
    평균 감성 점수에 따라 시각화 배지 문자열을 반환합니다.
    """
    if score >= 0.6:
        return "🟢 긍정적 (Positive)"
    elif score <= 0.4:
        return "🔴 부정적 (Negative)"
    return "🟡 중립적 (Neutral)"


def get_score_color_html(score: float) -> str:
    """
    개별 뉴스 감성 점수에 따라 색상 입혀진 HTML 문자열을 반환합니다.
    """
    if score >= 0.6:
        return f"<span style='color:#2ecc71; font-weight:bold;'>긍정 ({score:.4f})</span>"
    elif score <= 0.4:
        return f"<span style='color:#e74c3c; font-weight:bold;'>부정 ({score:.4f})</span>"
    return f"<span style='color:#7f8c8d; font-weight:bold;'>중립 ({score:.4f})</span>"


# ── 데이터 패치 함수 ────────────────────────────────────────────────────────────
def fetch_analysis_data(stock_name: str, limit: int) -> Dict[str, Any]:
    """
    FastAPI 백엔드 서버를 호출하여 종목 분석 데이터를 가져옵니다.
    
    Args:
        stock_name: 분석 대상 종목명
        limit: 뉴스 수집 건수 제한
        
    Returns:
        백엔드 서버로부터 받은 JSON 응답 딕셔너리
        
    Raises:
        httpx.HTTPError: API 통신 과정에서 오류 발생 시
    """
    params = {"stock_name": stock_name, "limit": limit}
    # 요약, 감성분석, Gemini 리포트 생성까지 시간이 걸리므로 timeout을 넉넉히 90초 설정
    with httpx.Client(timeout=90.0) as client:
        response = client.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()


# ── 메인 UI 그리기 ──────────────────────────────────────────────────────────────
def main() -> None:
    # 1. 페이지 설정
    st.set_page_config(
        page_title="주식 감성 분석 보조 시스템",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 커스텀 스타일링 주입 (폰트 및 여백 보완)
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # 메인 헤더
    st.markdown('<div class="main-title">📈 AI 기반 주식 뉴스 감성 분석 및 투자 보조 대시보드</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">네이버 실시간 뉴스 기사를 스크랩하고 KoBART의 요약, KoELECTRA의 감성 지수, Gemini의 종합 보고서를 한눈에 제공합니다.</div>', unsafe_allow_html=True)

    # 2. 사이드바 구성
    st.sidebar.header("🔍 분석 설정")
    stock_name = st.sidebar.text_input("분석 주식 종목명", value="삼성전자", help="분석할 기업명을 입력하세요. (예: 삼성전자, SK하이닉스)")
    limit = st.sidebar.slider("뉴스 수집 제한 개수", min_value=5, max_value=30, value=15, step=5, help="분석할 최근 네이버 뉴스 기사의 최대 개수를 지정합니다.")
    
    start_analysis = st.sidebar.button("⚡ 분석 시작", use_container_width=True)

    # 3. 분석 시작 버튼 이벤트 핸들링
    if start_analysis:
        if not stock_name.strip():
            st.error("종목명을 올바르게 입력해 주세요.")
            return

        # 백엔드 API 요청 중 스피너 표시
        with st.spinner("실시간 뉴스 크롤링 및 AI 분석 리포트를 생성 중입니다. 약 10~15초 소요됩니다..."):
            try:
                data = fetch_analysis_data(stock_name.strip(), limit)
                
                # 결과 파싱
                avg_score = data.get("avg_sentiment_score", 0.5)
                total_count = data.get("total_news_count", 0)
                news_list: List[Dict[str, Any]] = data.get("news_list", [])
                report_markdown = data.get("report", "")

                # 4. 메인 화면 - 종합 요약 메트릭 출력
                st.subheader("📊 감성 분석 종합 지표")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("검색 종목", stock_name.strip())
                with col2:
                    st.metric("분석 뉴스 수", f"{total_count} 건")
                with col3:
                    # 감성 배지와 함께 점수 출력
                    badge = get_sentiment_badge(avg_score)
                    st.metric("평균 감성 점수 (0.0~1.0)", f"{avg_score:.4f}", delta=badge, delta_color="off")

                st.markdown("---")

                # 5. 메인 화면 - AI 종합 보고서 출력 (메인 영역)
                st.subheader("🤖 AI 종합 투자 분석 보고서")
                if report_markdown:
                    st.markdown(report_markdown)
                else:
                    st.warning("생성된 AI 보고서가 없습니다.")

                st.markdown("---")

                # 6. 메인 화면 - 뉴스 세부 리스트 출력
                st.subheader("📰 뉴스 세부 분석 피드")
                if not news_list:
                    st.info("수집된 상세 뉴스 기사가 없습니다.")
                else:
                    for i, news in enumerate(news_list, start=1):
                        title = news.get("title", "(제목 없음)")
                        link = news.get("link", "#")
                        publisher = news.get("publisher", "미상")
                        date = news.get("date", "날짜 미상")
                        score = news.get("sentiment_score", 0.5)
                        summary = news.get("summary", "요약 정보 없음")

                        # 각 기사별 감성 강조 HTML 구성
                        score_html = get_score_color_html(score)
                        
                        # Expander 형태로 기사 피딩
                        header_text = f"[{publisher}] {title}"
                        with st.expander(header_text, expanded=(i == 1)): # 첫 번째 기사는 기본 펼침
                            st.markdown(f"**🔗 원문 기사**: [{title}]({link})")
                            st.markdown(f"**✍️ 언론사/날짜**: {publisher} | {date}")
                            st.markdown(f"**⚖️ 감성 분류**: {score_html}", unsafe_allow_html=True)
                            st.markdown("**📝 KoBART 3줄 요약**:")
                            st.info(summary)

            except httpx.HTTPError as http_err:
                st.error("백엔드 서버와 통신하는 과정에서 오류가 발생했습니다.")
                st.info(f"상세 에러: {http_err}")
                st.warning("FastAPI 백엔드 서버가 작동 중인지 확인해 주세요. (기본 포트: 8000)")
            except Exception as e:
                st.error("데이터 처리 중 예기치 못한 에러가 발생했습니다.")
                st.exception(e)
    else:
        # 서비스 안내 화면
        st.info("사이드바에서 종목명을 입력하고 '분석 시작' 버튼을 누르면 실시간 분석 대시보드가 로드됩니다.")
        
        # 헬스체크 동작 검증 (백엔드 서버 상태 모니터링)
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(HEALTH_CHECK_URL)
                if r.status_code == 200:
                    st.success("🟢 백엔드 분석 서버가 정상적으로 연결되어 작동 중입니다.")
                else:
                    st.warning("🟡 백엔드 서버 상태가 올바르지 않습니다.")
        except Exception:
            st.error("🔴 백엔드 서버에 연결할 수 없습니다. FastAPI 서버가 켜져 있는지 확인해 주세요.")

if __name__ == "__main__":
    main()
