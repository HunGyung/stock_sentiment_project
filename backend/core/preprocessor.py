import re

def clean_news_text(text: str) -> str:
    """
    뉴스 본문에서 이메일, 기자명, 언론사 정보, 무단 전재 금지 문구,
    그리고 불필요한 특수문자 및 공백을 정교하게 제거합니다.
    """
    if not text:
        return ""
        
    # 0. 앞뒤 공백 제거
    text = text.strip()
    
    # 1. HTML 엔티티 및 escape 문자 제거
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = text.replace('\xa0', ' ').replace('\u200b', ' ')
    
    # 2. 언론사 접두사 제거 (예: (서울=연합뉴스), (서울 연합뉴스), [서울=뉴스1], [서울=edaily])
    # \w는 유니코드 한글, 영문, 숫자를 모두 포함합니다. re.M 플래그로 줄 단위 매칭을 지원합니다.
    text = re.sub(r'^[\[\(][가-힣\s]+[=\s]+[\w\s\(\)\-\.]+[\]\)]', '', text, flags=re.M)
    # 기사 처음에 위치하는 "서울=연합뉴스" 패턴 제거
    text = re.sub(r'^[가-힣\s]+[=\s]+[\w\s\(\)\-\.]+뉴스\s*', '', text, flags=re.M)
    
    # 3. 사진 설명 및 불필요한 대괄호/괄호 영역 제거 (예: [사진 제공=삼성전자], [자료사진])
    text = re.sub(r'\[사진[^\]]*\]', '', text)
    text = re.sub(r'\[자료[^\]]*\]', '', text)
    text = re.sub(r'\(사진[^\)]*\)', '', text)
    text = re.sub(r'\(자료[^\)]*\)', '', text)
    text = re.sub(r'\[[^\]]*DB\]', '', text)
    text = re.sub(r'\[[^\]]*캡처\]', '', text)
    
    # 4. [기자명 + 이메일] 복합 패턴 제거 (이메일 단독 제거 전에 수행해야 함)
    # 예: "홍길동 기자 (gildong@news.com)", "정은지 기자(jeong@news.com)"
    text = re.sub(r'[가-힣]{2,4}\s*(기자|특파원|PD|pd|연구원|연구위원)\s*\(?[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\)?', '', text)
    
    # 5. 이메일 주소 단독 제거
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    
    # 6. 기자명 단독 패턴 제거
    # 예: "박재민 기자 = ", "김철수 기자"
    text = re.sub(r'[가-힣]{2,4}\s*(기자|특파원|PD|pd|연구원|연구위원|글)\s*=\s*', '', text)
    text = re.sub(r'[가-힣]{2,4}\s*(기자|특파원|PD|pd)\b', '', text)
    
    # 7. 네이버 구독/뉴스레터 링크 및 특수 기호로 시작하는 안내선 제거
    text = re.sub(r'▶\s*[^\n]+', '', text)
    text = re.sub(r'ⓒ\s*[^\n]+', '', text)
    
    # 8. 저작권 및 무단 전재 금지 문구 제거
    text = re.sub(r'Copyrights?\s*©?.*$', '', text, flags=re.IGNORECASE | re.M)
    text = re.sub(r'무단\s*전재\s*및?\s*재배포\s*금지', '', text)
    text = re.sub(r'재배포\s*금지', '', text)
    text = re.sub(r'저작권자\s*©?.*$', '', text, flags=re.M)
    
    # 9. 이메일/기자명 제거로 인해 남은 빈 괄호 제거
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # 10. 불필요한 특수문자 제거 (한글, 영문, 숫자, 주요 문장부호 .,!?%,%()만 남김)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s\.\,\!\?\%\(\)\-\+\w]', ' ', text)
    
    # 11. 연속된 공백 및 줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
