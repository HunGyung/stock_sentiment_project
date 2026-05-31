import os
import sys

# Avoid OpenMP runtime conflict error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# Add the project root directory to Python path for relative imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.preprocessor import clean_news_text


class KoBARTSummarizer:
    """Local KoBART-based abstractive summarizer for Korean news articles."""

    def __init__(self, model_name: str = "gogamza/kobart-summarization") -> None:
        """
        Load the KoBART summarization model and tokenizer.

        Args:
            model_name: Hugging Face model identifier or local model directory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Using device: {self.device}")

        try:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load summarization model '{model_name}'.") from exc

        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str, max_length: int = 128, min_length: int = 20) -> str:
        """
        Summarize a Korean news article into a concise abstract.

        Args:
            text: Raw Korean news article body.
            max_length: Maximum generated summary token length.
            min_length: Minimum generated summary token length.

        Returns:
            Generated summary text. Returns an empty string when input is empty.
        """
        if not text or not text.strip():
            return ""

        cleaned_text = clean_news_text(text)
        if not cleaned_text:
            return ""

        raw_input_ids = self.tokenizer.encode(
            cleaned_text,
            max_length=1022,
            truncation=True,
        )
        input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=input_tensor,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = summary.replace("\u2581", " ")
        return self._limit_to_three_lines(summary.strip())

    @staticmethod
    def _limit_to_three_lines(summary: str) -> str:
        """
        Keep a generated summary within three readable lines.

        Args:
            summary: Generated summary text.

        Returns:
            Summary text containing at most three non-empty lines.
        """
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        if len(lines) > 1:
            return "\n".join(lines[:3])

        sentences = [sentence.strip() for sentence in summary.split(".") if sentence.strip()]
        if len(sentences) > 1:
            return ". ".join(sentences[:3]) + "."

        if lines:
            return lines[0]

        if not sentences:
            return summary

        return sentences[0] + "."


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample_article = """
    네이버가 국방 AI 관련 사업을 전담하는 조직을 새롭게 개설한다.

    31일 정보기술(IT) 업계에 따르면 네이버클라우드는 6월1일 자로 국방 AX(AI 전환) 김유원 대표가 직접 이끄는 전담 태스크포스(TF)를 출범할 예정이다.

    네이버가 국방 AI 사업만을 전담하는 조직을 별도로 꾸리는 것은 처음이다.

    네이버클라우드는 이 조직을 통해 AI 모델 개발, 사업 개발, 홍보·마케팅 기능을 결합해 국방 특화 AI 모델 개발과 사업화를 추진할 계획이다.

    국방 AX TF는 군 현장에 직접 투입돼 부대별 맞춤형 AI 솔루션을 설계하고 구현하는 'FDE(현장 배치 엔지니어)' 직군을 핵심 전면에 배치한다.

    조직은 이를 중심으로 AI 모델 개발, 사업 개발, 홍보·마케팅 등 유관 부서들이 유기적으로 협업하는 형태로 운영될 방침이다.

    네이버클라우드가 국방 AI 전담 조직을 신설한 배경에는 글로벌 AI 기술 경쟁이 안보 영역으로 확대되면서, 국방 분야에서도 기술 자립과 데이터 주권 확보가 필수 조건이 됐다는 판단이 깔린 것으로 풀이된다.

    특히 군사 기밀이나 안보 핵심 정보 등 민감한 데이터를 다루는 국방의 특성상, 독자적으로 AI 서비스와 인프라를 구축·운영할 수 있는 역량이 필수적이라는 것이 업계의 분석이다.

    이에 따라 네이버클라우드는 자사가 보유한 독자적 기반 모델과 클라우드 인프라, 데이터 주권을 지키는 '소버린(Sovereign) AI' 역량을 동원해 대한민국 국방 안보 환경에 최적화된 특화 AI 솔루션을 공급하겠다는 구상이다.

    또한 텍스트와 이미지, 음성 등 여러 데이터를 결합해 추론 및 생성할 수 있는 옴니모달 AI 기술을 국방 분야에 적용하는 방안을 추진할 예정이다.

    업계에서는 네이버클라우드가 미국의 팔란티어 기업처럼 안보 특화형 AI 사업 모델을 구축할 수 있을지 여부에 주목하고 있다.

    팔란티어는 미국 정부와 국방부, 정보기관 등의 방대한 데이터를 분석·시각화하는 플랫폼을 공급하며 글로벌 기업으로 성장한 곳이다.

    네이버 관계자는 “네이버의 원천 AI 기술력을 바탕으로 국방 분야의 특수성을 반영한 맞춤형 솔루션을 지속적으로 고도화할 것”이라며 “국내 국방 AI 생태계의 경쟁력 강화와 질적 성장에 기여하겠다”고 말했다.
    """
    """
    삼성전자가 인공지능 반도체와 고대역폭메모리(HBM) 수요 확대에 대응하기 위해 차세대 메모리 생산라인 투자를
    앞당기기로 했다. 업계에 따르면 삼성전자는 주요 글로벌 클라우드 기업과 서버용 메모리 공급 협상을 진행하고
    있으며, 하반기부터 HBM3E와 DDR5 제품 출하량을 단계적으로 늘릴 계획이다. 증권가에서는 메모리 가격 반등과
    재고 조정 마무리로 반도체 부문 실적 개선 속도가 빨라질 수 있다고 보고 있다. 다만 파운드리 부문은 선단 공정
    고객 확보 경쟁이 이어지고 있어 단기간에 수익성이 크게 개선되기는 어렵다는 분석도 나온다. 삼성전자는 모바일,
    서버, 차량용 반도체를 아우르는 포트폴리오를 강화하고 연구개발 투자를 확대해 시장 변동성에 대응하겠다는
    방침이다. 한편 원달러 환율과 글로벌 경기 회복 속도, 미국과 중국의 기술 규제 변화는 향후 주가 흐름에 영향을
    줄 주요 변수로 꼽힌다. 전문가들은 단기 주가가 실적 기대감을 상당 부분 반영했지만, AI 서버 투자 사이클이
    장기화될 경우 중장기 성장 동력은 여전히 유효하다고 평가했다.
    """

    cleaned_article = clean_news_text(sample_article)
    print("\n--- Preprocessed Article ---")
    print(cleaned_article)

    print("\n--- KoBART Summary ---")
    summarizer = KoBARTSummarizer()
    print(summarizer.summarize(sample_article))
