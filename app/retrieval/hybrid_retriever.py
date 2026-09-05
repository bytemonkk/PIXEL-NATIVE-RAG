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

        bm25_results = self.bm25.search(
            query=query,
            top_k=top_k,
        )

        print()
        print("BM25 results inside HybridRetriever:")

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{result.tile_id} "
                f"score={result.score:.6f}"
            )

        raise NotImplementedError(
            "BM25 stage completed successfully."
        )