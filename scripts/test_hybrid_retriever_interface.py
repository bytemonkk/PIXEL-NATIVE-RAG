from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evidence_builder import EvidenceBuilder
from app.retrieval.faiss_index import FAISSVisualIndex
from app.retrieval.rrf import RRFFusion
from app.retrieval.visual_encoder import SigLIPEncoder
from app.retrieval.hybrid_retriever import HybridRetriever


def main():
    index_dir = Path(
        "data/index/shared_sample_document"
    )

    print()
    print("Loading dependencies...")

    bm25 = BM25Retriever.load(
        index_dir
    )

    faiss_index = FAISSVisualIndex.load(
        index_dir / "visual.index"
    )

    encoder = SigLIPEncoder()

    fusion = RRFFusion(
        k=60
    )

    evidence_builder = EvidenceBuilder(
        ocr_metadata_path=(
            index_dir / "ocr.json"
        ),
        visual_metadata_path=(
            index_dir / "visual_metadata.json"
        ),
    )

    print()
    print("Creating HybridRetriever...")

    retriever = HybridRetriever(
        bm25=bm25,
        faiss_index=faiss_index,
        encoder=encoder,
        fusion=fusion,
        evidence_builder=evidence_builder,
    )

    print()
    print(
        "HybridRetriever created successfully!"
    )

    print(
        "BM25 documents:",
        len(retriever.bm25.documents),
    )

    print(
        "FAISS vectors:",
        retriever.faiss_index.size,
    )

    print(
        "Encoder:",
        retriever.encoder.model_name,
    )

    print(
        "RRF k:",
        retriever.fusion.k,
    )

    print(
        "EvidenceBuilder:",
        type(
            retriever.evidence_builder
        ).__name__,
    )

    print()
    print(
        "Dependency injection test successful!"
    )
    print()


if __name__ == "__main__":
    main()