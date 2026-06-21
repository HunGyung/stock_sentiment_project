"""
backend/core/rag_service.py

로컬 벡터 DB(ChromaDB) 기반 RAG(Retrieval-Augmented Generation) 서비스 모듈.

뉴스 본문을 의미 단위 청크로 분할하고 SBERT로 임베딩하여 ChromaDB에 저장합니다.
보고서 생성 시 쿼리와 의미적으로 유사한 문맥(Context)을 Top-K 검색으로 반환합니다.

파이프라인:
  [뉴스 크롤링]
      → [본문 Chunking]         ← chunk_text()
      → [SBERT 임베딩 변환]     ← SentenceTransformer
      → [ChromaDB 적재]         ← StockRAGService.index_news()
      → [유사도 검색 (Top-K)]   ← StockRAGService.retrieve_contexts()
      → [Gemini 보고서 생성]
"""

import hashlib
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch

# ChromaDB, SentenceTransformers는 선택적 의존성으로 처리합니다.
# 미설치 환경에서도 임포트 오류 없이 서버가 기동될 수 있도록 지연 로드합니다.
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── 경로 상수 ───────────────────────────────────────────────────────────────────
# __file__ 기준 절대 경로로 계산하여 실행 디렉토리에 무관하게 동작합니다.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
_CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT, "data", "chromadb")

# ── 설정 상수 ───────────────────────────────────────────────────────────────────
COLLECTION_NAME = "stock_news_collection"
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"       # 확인된 한국어 SBERT (768-dim)
FALLBACK_EMBED_MODEL = "snunlp/KR-SBERT-V2-nli-sts"   # 폴백 모델
CHUNK_SIZE = 250          # 청크 최대 문자 수
CHUNK_OVERLAP = 50        # 연속 청크 간 겹침 문자 수
DEFAULT_TOP_K = 3         # 기본 검색 결과 개수


# ── 헬퍼 함수 ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    뉴스 본문을 일정한 크기(chunk_size)의 조각으로 분할합니다.

    분할 우선순위:
      1. 문장 종결 기호(., !, ?, 。) 기준으로 자연스럽게 분절
      2. 문장이 chunk_size를 초과하면 강제로 글자 수 기준 분할

    조각 간 문맥 연결을 위해 일부 텍스트가 겹치도록(overlap) 구성합니다.

    Args:
        text: 분할 대상 원문 텍스트
        chunk_size: 청크 최대 문자 수 (기본값: 250)
        overlap: 이전 청크와 겹치는 문자 수 (기본값: 50)

    Returns:
        비어 있지 않은 청크 문자열 리스트
    """
    if not text or not text.strip():
        return []

    # 1단계: 문장 종결 기호 기준으로 1차 분절
    sentences: List[str] = re.split(r"(?<=[.!?。])\s+", text.strip())

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 단일 문장 자체가 chunk_size를 초과하는 경우 강제 분할
        if len(sentence) > chunk_size:
            # 현재 청크가 있으면 먼저 저장
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # 긴 문장을 chunk_size 단위로 강제 분할
            for i in range(0, len(sentence), chunk_size - overlap):
                sub = sentence[i: i + chunk_size]
                if sub.strip():
                    chunks.append(sub.strip())
            continue

        # 문장을 추가했을 때 chunk_size를 초과하면 현재 청크를 저장하고 새로 시작
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # overlap: 직전 청크의 마지막 overlap자를 새 청크의 시작에 붙임
            current_chunk = current_chunk[-overlap:].strip() + " " + sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    # 남은 텍스트 처리
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


def _make_chunk_id(stock_name: str, article_index: int, chunk_index: int) -> str:
    """
    종목명 + 기사 인덱스 + 청크 인덱스 조합으로 결정론적 고유 ID를 생성합니다.

    ChromaDB의 upsert 키로 사용하여 중복 적재를 방지합니다.

    Args:
        stock_name: 종목명
        article_index: 기사 순번 (0-based)
        chunk_index: 청크 순번 (0-based)

    Returns:
        SHA-256 기반 16자리 hex 문자열 ID
    """
    raw = f"{stock_name}::{article_index}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ── RAG 서비스 클래스 ────────────────────────────────────────────────────────────

class StockRAGService:
    """
    ChromaDB + SBERT 기반 주식 뉴스 RAG 서비스.

    뉴스 본문을 청크로 분할 → SBERT로 임베딩 → ChromaDB에 저장하고,
    사용자 쿼리에 대한 의미 기반 유사 문맥 검색을 제공합니다.

    Attributes:
        embed_model (SentenceTransformer): 로컬 한국어 SBERT 임베딩 모델
        chroma_client (chromadb.PersistentClient): 로컬 영속성 ChromaDB 클라이언트
        collection (chromadb.Collection): 뉴스 청크 컬렉션 (코사인 유사도)
        device (str): 임베딩 연산 디바이스 ('cuda' 또는 'cpu')
    """

    def __init__(self) -> None:
        """
        SBERT 임베딩 모델과 ChromaDB 클라이언트를 초기화합니다.

        Raises:
            ImportError: chromadb 또는 sentence-transformers 미설치 시
            RuntimeError: 모델 로드 또는 ChromaDB 초기화 실패 시
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )
        if not SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

        # 임베딩 디바이스 설정 (GPU 우선)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[StockRAGService] Using device: %s", self.device)

        # SBERT 임베딩 모델 로드 (주요 모델 → 폴백 모델 순서로 시도)
        self.embed_model = self._load_embed_model()

        # ChromaDB 영속성 클라이언트 초기화
        os.makedirs(_CHROMA_PERSIST_DIR, exist_ok=True)
        logger.info(
            "[StockRAGService] ChromaDB persist dir: %s", _CHROMA_PERSIST_DIR
        )
        self.chroma_client = chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)

        # 컬렉션 로드 또는 생성 (코사인 유사도 기준)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "[StockRAGService] Collection '%s' ready. Current doc count: %d",
            COLLECTION_NAME,
            self.collection.count(),
        )

    def _load_embed_model(self) -> "SentenceTransformer":
        """
        SBERT 임베딩 모델을 로드합니다.

        주요 모델 로드에 실패할 경우 폴백 모델로 자동 전환합니다.

        Returns:
            로드된 SentenceTransformer 인스턴스

        Raises:
            RuntimeError: 주요·폴백 모델 모두 로드 실패 시
        """
        for model_name in [EMBED_MODEL_NAME, FALLBACK_EMBED_MODEL]:
            try:
                logger.info(
                    "[StockRAGService] Loading embedding model: %s", model_name
                )
                model = SentenceTransformer(model_name, device=self.device)
                logger.info(
                    "[StockRAGService] Embedding model loaded successfully: %s",
                    model_name,
                )
                return model
            except Exception as exc:
                logger.warning(
                    "[StockRAGService] Failed to load '%s': %s. Trying fallback...",
                    model_name,
                    exc,
                )

        raise RuntimeError(
            f"All embedding models failed to load. "
            f"Tried: [{EMBED_MODEL_NAME}, {FALLBACK_EMBED_MODEL}]"
        )

    # ── 데이터 적재 ─────────────────────────────────────────────────────────────

    def _clear_stock_data(self, stock_name: str) -> None:
        """
        특정 종목의 기존 ChromaDB 데이터를 삭제합니다.

        새로운 검색 세션마다 이전 데이터와 혼선을 방지하기 위해
        index_news() 호출 전 자동으로 실행됩니다.

        Args:
            stock_name: 삭제 대상 종목명
        """
        try:
            existing = self.collection.get(
                where={"stock_name": stock_name},
                include=[],  # ID만 조회
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    "[StockRAGService] Cleared %d existing chunks for stock='%s'",
                    len(ids_to_delete),
                    stock_name,
                )
        except Exception as exc:
            logger.warning(
                "[StockRAGService] Failed to clear existing data for stock='%s': %s",
                stock_name,
                exc,
            )

    def index_news(
        self,
        news_list: List[Dict[str, Any]],
        stock_name: str,
    ) -> int:
        """
        크롤링된 뉴스 리스트를 청킹·임베딩하여 ChromaDB에 적재합니다.

        처리 흐름:
          1. 해당 종목의 기존 데이터 초기화 (격리 보장)
          2. 각 뉴스 본문을 chunk_text()로 분할
          3. 배치 SBERT 임베딩 생성
          4. ChromaDB에 ID, 벡터, 메타데이터 upsert

        Args:
            news_list: 뉴스 딕셔너리 리스트.
                       각 딕셔너리는 'title'(str), 'body'(str), 'date'(str),
                       'publisher'(str) 키를 포함해야 합니다.
            stock_name: 종목명 (메타데이터 필터링용)

        Returns:
            ChromaDB에 적재된 총 청크 수

        Raises:
            ValueError: news_list가 비어 있거나 stock_name이 공백인 경우
        """
        if not news_list:
            raise ValueError("news_list must not be empty.")
        if not stock_name or not stock_name.strip():
            raise ValueError("stock_name must not be empty or whitespace.")

        stock_name = stock_name.strip()

        # 기존 데이터 초기화 (종목별 격리)
        self._clear_stock_data(stock_name)

        all_chunks: List[str] = []
        all_ids: List[str] = []
        all_metadatas: List[Dict[str, str]] = []

        for article_idx, news in enumerate(news_list):
            title = news.get("title", "").strip()
            body = news.get("body", "").strip()
            date = news.get("date", "")
            publisher = news.get("publisher", "")
            link = news.get("link", "")

            # 본문이 없으면 제목으로 대체하여 최소한의 정보를 보존
            source_text = body if body else title
            if not source_text:
                logger.debug(
                    "[StockRAGService] Skipping article %d (no content)", article_idx
                )
                continue

            chunks = chunk_text(source_text)
            if not chunks:
                continue

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = _make_chunk_id(stock_name, article_idx, chunk_idx)
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadatas.append(
                    {
                        "stock_name": stock_name,
                        "article_index": str(article_idx),
                        "chunk_index": str(chunk_idx),
                        "title": title[:200],   # ChromaDB 메타데이터 길이 제한 대비 트리밍
                        "date": date,
                        "publisher": publisher,
                        "link": link,
                    }
                )

        if not all_chunks:
            logger.warning(
                "[StockRAGService] No valid chunks generated for stock='%s'", stock_name
            )
            return 0

        # 배치 임베딩 생성 (GPU 가속 적용)
        logger.info(
            "[StockRAGService] Encoding %d chunks for stock='%s'...",
            len(all_chunks),
            stock_name,
        )
        embeddings = self.embed_model.encode(
            all_chunks,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # ChromaDB upsert (ID 충돌 시 덮어쓰기)
        self.collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )

        logger.info(
            "[StockRAGService] Indexed %d chunks for stock='%s' into ChromaDB.",
            len(all_chunks),
            stock_name,
        )
        return len(all_chunks)

    # ── 컨텍스트 검색 ───────────────────────────────────────────────────────────

    def retrieve_contexts(
        self,
        query: str,
        stock_name: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[str]:
        """
        쿼리와 종목명을 기준으로 가장 유사한 뉴스 텍스트 조각을 검색합니다.

        종목명 메타데이터 필터를 사용해 타 종목 데이터의 오염을 방지합니다.

        Args:
            query: 의미 검색 쿼리 문자열
                   (예: "최근 실적 및 호재", "시장 리스크 및 부정 요인")
            stock_name: 검색 대상 종목명 (메타데이터 필터링용)
            top_k: 반환할 최대 컨텍스트 수 (기본값: 3)

        Returns:
            유사도 내림차순으로 정렬된 텍스트 청크 리스트.
            검색 실패 또는 결과 없음 시 빈 리스트를 반환합니다.

        Raises:
            ValueError: query 또는 stock_name이 비어 있는 경우
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace.")
        if not stock_name or not stock_name.strip():
            raise ValueError("stock_name must not be empty or whitespace.")

        stock_name = stock_name.strip()

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embed_model.encode(
                query.strip(),
                convert_to_numpy=True,
            ).tolist()

            # ChromaDB 유사도 검색 (종목 메타데이터 필터 적용)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"stock_name": stock_name},
                include=["documents", "distances", "metadatas"],
            )

            documents: List[List[str]] = results.get("documents", [[]])
            if not documents or not documents[0]:
                logger.info(
                    "[StockRAGService] No results found for query='%s', stock='%s'",
                    query,
                    stock_name,
                )
                return []

            contexts = documents[0]
            logger.info(
                "[StockRAGService] Retrieved %d context(s) for query='%s', stock='%s'",
                len(contexts),
                query,
                stock_name,
            )
            return contexts

        except Exception as exc:
            logger.error(
                "[StockRAGService] Context retrieval failed for query='%s', stock='%s': %s",
                query,
                stock_name,
                exc,
                exc_info=True,
            )
            return []

    def get_collection_count(self) -> int:
        """
        현재 ChromaDB 컬렉션에 저장된 총 청크 수를 반환합니다.

        Returns:
            총 청크 수 (int)
        """
        try:
            return self.collection.count()
        except Exception as exc:
            logger.warning(
                "[StockRAGService] Failed to get collection count: %s", exc
            )
            return 0


# ── 모듈 단독 실행 시 동작 테스트 ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(_PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("\n=== StockRAGService 초기화 테스트 ===")
    rag = StockRAGService()
    print(f"초기 컬렉션 청크 수: {rag.get_collection_count()}")

    # 더미 뉴스 데이터로 적재 테스트
    dummy_news = [
        {
            "title": "삼성전자, 2분기 영업이익 10조 돌파 전망",
            "body": (
                "삼성전자가 2분기 영업이익 10조 원을 돌파할 것으로 전망된다. "
                "반도체 업황 회복과 HBM 수요 급증이 주된 요인으로 꼽힌다. "
                "특히 AI 서버용 고대역폭 메모리 공급이 크게 늘면서 실적 개선을 견인하고 있다. "
                "증권가에서는 하반기 추가 상승 여력도 충분하다고 분석한다."
            ),
            "date": "2026-06-19T10:00:00+09:00",
            "publisher": "한국경제",
            "link": "https://example.com/news/1",
        },
        {
            "title": "미·중 무역 갈등 재확산…반도체 수출 제한 우려",
            "body": (
                "미국과 중국 간 무역 갈등이 재점화되며 국내 반도체 기업들의 대중 수출에 "
                "제동이 걸릴 수 있다는 우려가 커지고 있다. "
                "미국 정부는 추가 수출 통제 품목을 검토 중이며, "
                "삼성전자와 SK하이닉스가 주요 영향권 안에 들 수 있다고 업계는 분석한다. "
                "중국 매출 비중이 높은 기업일수록 리스크가 크다는 지적이 나온다."
            ),
            "date": "2026-06-18T14:00:00+09:00",
            "publisher": "조선비즈",
            "link": "https://example.com/news/2",
        },
    ]

    print("\n=== index_news 테스트 (삼성전자) ===")
    count = rag.index_news(dummy_news, stock_name="삼성전자")
    print(f"적재된 청크 수: {count}")
    print(f"컬렉션 총 청크 수: {rag.get_collection_count()}")

    print("\n=== retrieve_contexts 테스트 ===")
    queries = [
        "최근 실적 및 호재",
        "시장 리스크 및 부정 요인",
    ]
    for q in queries:
        print(f"\n[쿼리] {q}")
        contexts = rag.retrieve_contexts(q, stock_name="삼성전자", top_k=2)
        for i, ctx in enumerate(contexts, 1):
            print(f"  [{i}] {ctx[:80]}...")
