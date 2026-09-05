import json
from pathlib import Path
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
        visual_metadata_path: Path,
    ):
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.encoder = encoder
        self.fusion = fusion
        self.evidence_builder = evidence_builder

        self.visual_metadata_path = (
            visual_metadata_path
        )

        self.visual_metadata = (
            self._load_visual_metadata()
        )

    def _load_visual_metadata(self) -> dict:
        if not self.visual_metadata_path.exists():
            raise FileNotFoundError(
                "Visual metadata not found: "
                f"{self.visual_metadata_path}"
            )

        with self.visual_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)
        return {
            record["vector_id"]: record
            for record in records
        }
        
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EvidenceItem]:

        # --------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Visual retrieval
        # --------------------------------------------------

        query_vector = self.encoder.encode_text(
            query
        )

        print()
        print(
            "SigLIP query vector dimensions:",
            len(query_vector),
        )

        visual_results = self.faiss_index.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        print()
        print("Visual results inside HybridRetriever:")

        visual_tile_ids = []

        for rank, result in enumerate(
            visual_results,
            start=1,
        ):
            if result.vector_id not in self.visual_metadata:
                raise KeyError(
                    "Visual metadata missing for "
                    f"vector ID: {result.vector_id}"
                )

            metadata = self.visual_metadata[
                result.vector_id
            ]

            tile_id = metadata["tile_id"]

            visual_tile_ids.append(
                tile_id
            )

            print(
                f"  {rank}. "
                f"vector_id={result.vector_id} "
                f"tile_id={tile_id} "
                f"score={result.score:.6f}"
            )

        print()
        print("Visual tile IDs:")

        for rank, tile_id in enumerate(
            visual_tile_ids,
            start=1,
        ):
            print(
                f"  {rank}. {tile_id}"
            )

        # --------------------------------------------------
        # RRF fusion
        # --------------------------------------------------

        bm25_tile_ids = [
            result.tile_id
            for result in bm25_results
        ]

        print()
        print("Running RRF fusion inside HybridRetriever...")

        fused_results = self.fusion.fuse(
            ranked_lists={
                "bm25": bm25_tile_ids,
                "visual": visual_tile_ids,
            }
        )

        print()
        print("RRF results inside HybridRetriever:")

        for rank, result in enumerate(
            fused_results,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{result.tile_id} "
                f"score={result.score:.6f} "
                f"ranks={result.ranks}"
            )

        raise NotImplementedError(
            "RRF fusion completed successfully."
        )
                
