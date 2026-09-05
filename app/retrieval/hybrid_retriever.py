from app.retrieval.evidence import EvidenceItem


class HybridRetriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceItem]:
        raise NotImplementedError