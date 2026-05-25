# 뉴스 감성 분석 기반 주식 투자 보조 시스템 (Stock Sentiment Assistant)

자연어 처리(NLP) 기술을 활용하여 주식 뉴스를 실시간으로 분석하고 투자자에게 객관적인 감성 지표를 제공하는 엔드투엔드(End-to-End) 웹 서비스입니다. 

로컬 경량 AI 모델(KoELECTRA, KoBART)과 클라우드 거대 언어 모델(Gemini 2.5 Flash)을 결합한 **하이브리드 추론 아키텍처**로 설계되었습니다.

---

## 📂 프로젝트 구조 (Directory Structure)

```text
stock_sentiment_project/
├── backend/
│   ├── core/
│   │   ├── scraper.py       # 네이버 뉴스 비동기 크롤러
│   │   └── preprocessor.py  # 정규표현식 기반 텍스트 정제기
│   └── models/
│       ├── train_sentiment.py # KoELECTRA 파인튜닝 학습 스크립트
│       └── sentiment.py       # KoELECTRA 감성 분석 추론 모듈
├── data/
│   ├── finance_data.csv       # 학습용 한국어 금융 뉴스 감성 데이터셋
│   └── model_save/            # 학습 완료된 베스트 모델 가중치 및 설정
├── requirements.txt           # 프로젝트 의존성 라이브러리 명세
└── README.md                  # 프로젝트 설명 및 진행 문서 (본 파일)
```

---

## 📈 개발 진행 상황 (Development Progress)

### 1. 가상환경 및 GPU 가속 환경 세팅 (완료)
* Python 3.13 및 Windows 환경에 맞추어 PyTorch CUDA 12.4 버전을 설치하여 로컬 GPU 가속 연동을 확보하였습니다.

### 2. 뉴스 수집 및 정제 파이프라인 (완료)
* **스크래퍼 ([scraper.py](backend/core/scraper.py))**: `httpx`와 `BeautifulSoup4`를 이용해 네이버 뉴스 검색 결과를 실시간 비동기로 긁어오는 파이프라인 구축.
* **전처리기 ([preprocessor.py](backend/core/preprocessor.py))**: 기사 내 이메일, 기자명, 언론사 정보, 저작권 문구 등 노이즈를 완벽하게 지워내는 정규표현식 클리너 함수 구현.

### 3. 로컬 감성 분석 모델 개발 (완료)
* **데이터셋**: 글로벌 표준인 *Financial PhraseBank*의 한국어 검수 버전(`finance_data.csv`, 4,846문장) 활용.
* **학습 기법 ([train_sentiment.py](backend/models/train_sentiment.py))**:
  * 중립 데이터 편향(59.4%)을 해결하기 위해 `WeightedRandomSampler` 도입.
  * 범용 언어 지식 유실 방지 및 리소스 아키텍처 최적화를 위한 **하위 3개 레이어 동결(Layer Freezing)** 적용.
  * 8에폭 학습 결과 **검증 정확도 81.03%**, **Macro F1-Score 0.7883** 달성하여 최적의 수렴점에 도달.
* **추론 모듈 ([sentiment.py](backend/models/sentiment.py))**:
  * 학습 완료된 커스텀 가중치 모델을 로드하여 문장의 긍정/부정/중립 확률 도출.
  * 3클래스 확률 분포를 0.0 ~ 1.0 사이의 감성 스칼라 점수로 결합하는 공식 적용:
    $$\text{Score} = 0.5 \times P_{\text{neutral}} + P_{\text{positive}}$$

---

## 🚀 실행 및 테스트 방법 (How to Run)

### 1. 의존성 패키지 설치
가상환경 활성화 후 실행:
```bash
pip install -r requirements.txt
pip install scikit-learn
```

### 2. 감성 분석 모델 학습 실행
```bash
python backend/models/train_sentiment.py
```

### 3. 감성 분석 추론 모듈 테스트
```bash
python backend/models/sentiment.py
```

---

## 📝 향후 개발 계획 (Next Steps)
1. **KoBART 생성 요약 모델 개발 (`backend/models/summarizer.py`)**
   * 긴 기사 본문을 3줄 이내로 핵심 요약하는 로컬 AI 추론 클래스 구현.
2. **Gemini API 연동 클라이언트 구현 (`backend/models/gemini_client.py`)**
   * 감성 점수가 극단값($S \ge 0.85$ 또는 $S \le 0.15$)을 보일 때 투자 분석 리포트를 생성하는 LLM 프롬프팅 연동.
3. **FastAPI 백엔드 API 서버 구축 (`backend/api/main.py`)**
4. **Streamlit 웹 UI 시각화 대시보드 연동 (`frontend/app.py`)**
