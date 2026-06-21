# 1. 가볍고 안정적인 공식 파이썬 3.10 이미지 사용
FROM python:3.10-slim

# 2. 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 패키지 설치용 시스템 라이브러리 업데이트 (gcc 등 빌드 도구)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 패키지 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 백엔드 코드 및 데이터 모델 복사
COPY backend/ ./backend/
COPY data/ ./data/

# 6. PYTHONPATH에 루트 디렉토리 설정 (상대 임포트 에러 방지)
ENV PYTHONPATH=/app

# 7. 외부 요청을 받을 포트 노출 (FastAPI 기본: 8000)
EXPOSE 8000

# 8. FastAPI uvicorn 서버 실행
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]