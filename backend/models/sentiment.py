import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# add the project root directory to Python path for relative path recognition
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.preprocessor import clean_news_text

class KoELECTRASentimentAnalyzer:
    def __init__(self, model_dir: str = None):
        """
        loads the trained KoELECTRA Sentiment analysis model and its tokenizer
        """
        # 실행 디바이스 자동 감지 (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Using device: {self.device}")

        # ── [핵심 개선] 실행 디렉토리와 관계없이 절대 경로 계산 ────────────────────
        # __file__: backend/models/sentiment.py
        # current_dir: .../backend/models/
        # project_root: .../stock_sentiment_project/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        if model_dir is None:
            model_dir = os.path.join(project_root, "data", "model_save")
        # ──────────────────────────────────────────────────────────────────────────

        # 학습 가중치 로드
        if os.path.exists(model_dir):
            print(f"[{self.__class__.__name__}] Loading fine-tuned model from '{model_dir}'")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            # 절대 경로로도 못 찾았을 경우의 폴백
            print(f"[{self.__class__.__name__}] [WARNING] Fine-tuned model not found at '{model_dir}'. Loading base model.")
            default_model = "monologg/koelectra-base-v3-discriminator"
            self.tokenizer = AutoTokenizer.from_pretrained(default_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(default_model, num_labels=3)
            
        self.model.to(self.device)
        self.model.eval()  # sets Evaluation(Inference) Mode (disable Dropout, etc.)

    def analyze(self, text: str) -> float:
        """
        입력 문장을 정제한 뒤, 0.0(매우 부정) ~ 1.0(매우 긍정) 사이의 감성 스칼라 점수를 반환합니다.
        
        점수 계산 공식:
        Score = (0.0 * 부정_확률) + (0.5 * 중립_확률) + (1.0 * 긍정_확률)
              = (0.5 * 중립_확률) + 긍정_확률
        """
        if not text or not text.strip():
            return 0.5  # treates empty sentences as Neutral (0.5).
            
        # preprocessing the texts (removes unnecessary words)
        cleaned_text = clean_news_text(text)
        if not cleaned_text.strip():
            return 0.5
            
        # tokenizing and move to device (gpu)
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        ).to(self.device)
        
        # model inference (disable backpropagation)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [실수_값_0(부정), 실수_값_1(중립), 실수_값_2(긍정)]

        # transforms to a probablity distribution with a sum of 1.0 appliying Softmax
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [부정_확률, 중립_확률, 긍정_확률]
        
        prob_neg = probs[0].item()
        prob_neu = probs[1].item()
        prob_pos = probs[2].item()
        
        # calculate the scalar sentiment score (0.0 ~ 1.0)
        # return the average score assinging weight of 0 for Negative, 0.5 for Neutral, 1.0 for Positive
        sentiment_score = (0.5 * prob_neu) + prob_pos
        
        return sentiment_score

# test code
if __name__ == "__main__":
    analyzer = KoELECTRASentimentAnalyzer()
    
    test_sentences = [
        "삼성전자, 역대급 실적 발표에 주가 급등 상한가 기록!",
        "오늘 주식 시장은 별다른 소식 없이 보합세로 마감했습니다.",
        "글로벌 경기 침체 우려로 인해 외국인들이 주식을 대거 매도하며 주가가 급락했습니다.",
        "영화관株 '코로나 빙하기' 언제 끝나나…\"CJ CGV 올 4000억 손실 날수도\"",
        "현대제철, 지난해 영업익 3,313억원···전년比 67.7% 감소",
        "부품 공급 차질에…기아차 광주공장 전면 가동 중단",
        "C쇼크에 멈춘 흑자비행…대한항공 1분기 영업적자 566억",
        "'1000억대 횡령·배임' 최신원 회장 구속… SK네트웍스 \"경영 공백 방지 최선\""
    ]
    
    print("\n--- 추론 테스트 시작 ---")
    for sent in test_sentences:
        score = analyzer.analyze(sent)
        print(f"문장: {sent}")
        print(f"감성 점수: {score:.4f}")
        print("-" * 50)