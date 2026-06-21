# 1. 파이썬 3.10 이미지 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프론트엔드 코드 복사
COPY frontend/ ./frontend/

# 5. Streamlit 기본 포트 노출 (8501)
EXPOSE 8501

# 6. Streamlit 구동 명령어 실행
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]