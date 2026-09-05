from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evidence import EvidenceItem
from app.retrieval.evidence_builder import EvidenceBuilder
from app.retrieval.faiss_index import FAISSVisualIndex
from app.retrieval.rrf import RRFFusion
from app.retrieval.visual_encoder import SigLIPEncoder


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        faiss_index: FAISSVisualIndex,
        encoder: SigLIPEncoder,
        fusion: RRFFusion,
        evidence_builder: EvidenceBuilder,
    ):
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.encoder = encoder
        self.fusion = fusion
        self.evidence_builder = evidence_builder

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceItem]:
        raise NotImplementedError