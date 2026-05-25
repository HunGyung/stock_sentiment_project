import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 상대 경로 인식을 위해 프로젝트 루트 디렉토리를 파이썬 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.preprocessor import clean_news_text

class KoELECTRASentimentAnalyzer:
    def __init__(self, model_dir: str = "data/model_save"):
        """
        학습된 KoELECTRA 감성 분석 모델과 토크나이저를 로드합니다.
        """
        # 실행 디바이스 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Using device: {self.device}")
        
        # 학습 완료된 가중치 불러오기
        if os.path.exists(model_dir):
            print(f"[{self.__class__.__name__}] Loading fine-tuned model from '{model_dir}'")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            # 학습 폴더가 없을 경우를 대비한 폴백 (기본 사전학습 모델 로드 - 미학습 상태)
            print(f"[{self.__class__.__name__}] [WARNING] Fine-tuned model not found at '{model_dir}'. Loading base model.")
            default_model = "monologg/koelectra-small-v3-discriminator"
            self.tokenizer = AutoTokenizer.from_pretrained(default_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(default_model, num_labels=3)
            
        self.model.to(self.device)
        self.model.eval()  # 평가(추론) 모드 설정 (Dropout 등 비활성화)

    def analyze(self, text: str) -> float:
        """
        입력 문장을 정제한 뒤, 0.0(매우 부정) ~ 1.0(매우 긍정) 사이의 감성 스칼라 점수를 반환합니다.
        
        점수 계산 공식:
        Score = (0.0 * 부정_확률) + (0.5 * 중립_확률) + (1.0 * 긍정_확률)
              = (0.5 * 중립_확률) + 긍정_확률
        """
        if not text or not text.strip():
            return 0.5  # 빈 문장은 중립(0.5) 처리
            
        # 1. 텍스트 전처리 (우리가 만든 Regex 청소기 작동)
        cleaned_text = clean_news_text(text)
        if not cleaned_text.strip():
            return 0.5
            
        # 2. 토크나이징 및 디바이스 이동
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        ).to(self.device)
        
        # 3. 모델 추론 (기울기 계산 비활성화)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [실수_값_0(부정), 실수_값_1(중립), 실수_값_2(긍정)]
            
        # 4. Softmax를 취해 합이 1.0인 확률 분포로 변환
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [부정_확률, 중립_확률, 긍정_확률]
        
        prob_neg = probs[0].item()
        prob_neu = probs[1].item()
        prob_pos = probs[2].item()
        
        # 5. 스칼라 감성 점수 계산 (0.0 ~ 1.0)
        # 부정은 0점, 중립은 0.5점, 긍정은 1점으로 가중치를 주어 평균 점수 산출
        sentiment_score = (0.5 * prob_neu) + prob_pos
        
        return sentiment_score

# 간단한 모듈 독립 테스트용 코드
if __name__ == "__main__":
    analyzer = KoELECTRASentimentAnalyzer()
    
    test_sentences = [
        "삼성전자, 역대급 실적 발표에 주가 급등 상한가 기록!",
        "오늘 주식 시장은 별다른 소식 없이 보합세로 마감했습니다.",
        "글로벌 경기 침체 우려로 인해 외국인들이 주식을 대거 매도하며 주가가 급락했습니다."
    ]
    
    print("\n--- 추론 테스트 시작 ---")
    for sent in test_sentences:
        score = analyzer.analyze(sent)
        print(f"문장: {sent}")
        print(f"감성 점수: {score:.4f}")
        print("-" * 50)