import json
from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evidence_builder import EvidenceBuilder
from app.retrieval.faiss_index import FAISSVisualIndex
from app.retrieval.rrf import RRFFusion
from app.retrieval.visual_encoder import SigLIPEncoder


def main():
    index_dir = Path(
        "data/index/shared_sample_document"
    )

    query = "experimental results"
    retrieval_top_k = 5
    evidence_top_k = 5

    print()
    print("=" * 80)
    print("END-TO-END HYBRID RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Load BM25
    # --------------------------------------------------

    print()
    print("Step 1: Loading BM25 index...")

    bm25 = BM25Retriever.load(
        index_dir
    )

    print(
        f"BM25 documents: {len(bm25.documents)}"
    )

    # --------------------------------------------------
    # 2. BM25 retrieval
    # --------------------------------------------------

    print()
    print("Step 2: Running BM25...")
    print(f"Query: {query}")

    bm25_results = bm25.search(
        query=query,
        top_k=retrieval_top_k,
    )

    bm25_tile_ids = [
        result.tile_id
        for result in bm25_results
    ]

    print()
    print("BM25 ranking:")

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
    # 3. Load visual metadata
    # --------------------------------------------------

    print()
    print("Step 3: Loading visual metadata...")

    visual_metadata_path = (
        index_dir / "visual_metadata.json"
    )

    with visual_metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        visual_metadata = json.load(file)

    metadata_by_vector_id = {
        item["vector_id"]: item
        for item in visual_metadata
    }

    print(
        f"Visual metadata records: "
        f"{len(visual_metadata)}"
    )

    # --------------------------------------------------
    # 4. Load FAISS
    # --------------------------------------------------

    print()
    print("Step 4: Loading FAISS index...")

    faiss_index = FAISSVisualIndex.load(
        index_dir / "visual.index"
    )

    print(
        f"FAISS vectors: {faiss_index.size}"
    )

    # --------------------------------------------------
    # 5. Load SigLIP
    # --------------------------------------------------

    print()
    print("Step 5: Loading SigLIP...")

    encoder = SigLIPEncoder()

    # --------------------------------------------------
    # 6. Encode query
    # --------------------------------------------------

    print()
    print("Step 6: Encoding query...")

    query_vector = encoder.encode_text(
        query
    )

    print(
        f"Query vector dimensions: "
        f"{len(query_vector)}"
    )

    # --------------------------------------------------
    # 7. Visual retrieval
    # --------------------------------------------------

    print()
    print("Step 7: Running visual retrieval...")

    visual_results = faiss_index.search(
        query_vector=query_vector,
        top_k=retrieval_top_k,
    )

    visual_tile_ids = []

    print()
    print("Visual ranking:")

    for rank, result in enumerate(
        visual_results,
        start=1,
    ):
        metadata = metadata_by_vector_id[
            result.vector_id
        ]

        tile_id = metadata["tile_id"]

        visual_tile_ids.append(
            tile_id
        )

        print(
            f"  {rank}. "
            f"{tile_id} "
            f"score={result.score:.6f}"
        )

    # --------------------------------------------------
    # 8. RRF fusion
    # --------------------------------------------------

    print()
    print("Step 8: Running RRF fusion...")

    fusion = RRFFusion(
        k=60
    )

    fused_results = fusion.fuse(
        ranked_lists={
            "bm25": bm25_tile_ids,
            "visual": visual_tile_ids,
        }
    )

    print(
        f"Fused results: "
        f"{len(fused_results)}"
    )

    # --------------------------------------------------
    # 9. Select top-K evidence
    # --------------------------------------------------

    top_evidence_results = (
        fused_results[:evidence_top_k]
    )

    print()
    print(
        f"Step 9: Building top "
        f"{len(top_evidence_results)} evidence items..."
    )

    evidence_builder = EvidenceBuilder(
        ocr_metadata_path=(
            index_dir / "ocr.json"
        ),
        visual_metadata_path=(
            index_dir / "visual_metadata.json"
        ),
    )

    evidence = evidence_builder.build(
        top_evidence_results
    )

    # --------------------------------------------------
    # 10. Display final evidence
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL EVIDENCE")
    print("=" * 80)

    for rank, item in enumerate(
        evidence,
        start=1,
    ):
        print()
        print(f"Evidence Rank: {rank}")
        print(f"Tile ID: {item.tile_id}")
        print(f"Page: {item.page_number}")
        print(
            f"RRF Score: "
            f"{item.rrf_score:.6f}"
        )
        print(
            f"BM25 Rank: "
            f"{item.bm25_rank}"
        )
        print(
            f"Visual Rank: "
            f"{item.visual_rank}"
        )
        print(
            f"Image Exists: "
            f"{item.image_path.exists()}"
        )

        preview = (
            item.text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 200:
            preview = preview[:200] + "..."

        print(
            f"OCR Preview: {preview}"
        )

    # --------------------------------------------------
    # 11. Validation
    # --------------------------------------------------

    print()
    print("Step 10: Validating pipeline...")

    if len(evidence) != evidence_top_k:
        raise AssertionError(
            "Unexpected number of evidence items."
        )

    for item in evidence:

        if not item.image_path.exists():
            raise AssertionError(
                f"Evidence image missing: "
                f"{item.image_path}"
            )

        if not item.text.strip():
            raise AssertionError(
                f"OCR text is empty for "
                f"{item.tile_id}"
            )

        if item.rrf_score <= 0:
            raise AssertionError(
                f"Invalid RRF score for "
                f"{item.tile_id}"
            )

    if evidence[0].tile_id != (
        "shared_sample_document-p0004-t0002"
    ):
        raise AssertionError(
            "Unexpected top evidence tile."
        )

    print()
    print(
        "End-to-end retrieval test successful!"
    )
    print()
    print(
        "Query -> BM25 -> SigLIP/FAISS -> "
        "RRF -> EvidenceBuilder"
    )
    print(
        "All components communicated successfully."
    )
    print()


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evidence_builder import EvidenceBuilder
from app.retrieval.faiss_index import FAISSVisualIndex
from app.retrieval.rrf import RRFFusion
from app.retrieval.visual_encoder import SigLIPEncoder


def main():
    index_dir = Path(
        "data/index/shared_sample_document"
    )

    query = "experimental results"
    retrieval_top_k = 5
    evidence_top_k = 5

    print()
    print("=" * 80)
    print("END-TO-END HYBRID RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Load BM25
    # --------------------------------------------------

    print()
    print("Step 1: Loading BM25 index...")

    bm25 = BM25Retriever.load(
        index_dir
    )

    print(
        f"BM25 documents: {len(bm25.documents)}"
    )

    # --------------------------------------------------
    # 2. BM25 retrieval
    # --------------------------------------------------

    print()
    print("Step 2: Running BM25...")
    print(f"Query: {query}")

    bm25_results = bm25.search(
        query=query,
        top_k=retrieval_top_k,
    )

    bm25_tile_ids = [
        result.tile_id
        for result in bm25_results
    ]

    print()
    print("BM25 ranking:")

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
    # 3. Load visual metadata
    # --------------------------------------------------

    print()
    print("Step 3: Loading visual metadata...")

    visual_metadata_path = (
        index_dir / "visual_metadata.json"
    )

    with visual_metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        visual_metadata = json.load(file)

    metadata_by_vector_id = {
        item["vector_id"]: item
        for item in visual_metadata
    }

    print(
        f"Visual metadata records: "
        f"{len(visual_metadata)}"
    )

    # --------------------------------------------------
    # 4. Load FAISS
    # --------------------------------------------------

    print()
    print("Step 4: Loading FAISS index...")

    faiss_index = FAISSVisualIndex.load(
        index_dir / "visual.index"
    )

    print(
        f"FAISS vectors: {faiss_index.size}"
    )

    # --------------------------------------------------
    # 5. Load SigLIP
    # --------------------------------------------------

    print()
    print("Step 5: Loading SigLIP...")

    encoder = SigLIPEncoder()

    # --------------------------------------------------
    # 6. Encode query
    # --------------------------------------------------

    print()
    print("Step 6: Encoding query...")

    query_vector = encoder.encode_text(
        query
    )

    print(
        f"Query vector dimensions: "
        f"{len(query_vector)}"
    )

    # --------------------------------------------------
    # 7. Visual retrieval
    # --------------------------------------------------

    print()
    print("Step 7: Running visual retrieval...")

    visual_results = faiss_index.search(
        query_vector=query_vector,
        top_k=retrieval_top_k,
    )

    visual_tile_ids = []

    print()
    print("Visual ranking:")

    for rank, result in enumerate(
        visual_results,
        start=1,
    ):
        metadata = metadata_by_vector_id[
            result.vector_id
        ]

        tile_id = metadata["tile_id"]

        visual_tile_ids.append(
            tile_id
        )

        print(
            f"  {rank}. "
            f"{tile_id} "
            f"score={result.score:.6f}"
        )

    # --------------------------------------------------
    # 8. RRF fusion
    # --------------------------------------------------

    print()
    print("Step 8: Running RRF fusion...")

    fusion = RRFFusion(
        k=60
    )

    fused_results = fusion.fuse(
        ranked_lists={
            "bm25": bm25_tile_ids,
            "visual": visual_tile_ids,
        }
    )

    print(
        f"Fused results: "
        f"{len(fused_results)}"
    )

    # --------------------------------------------------
    # 9. Select top-K evidence
    # --------------------------------------------------

    top_evidence_results = (
        fused_results[:evidence_top_k]
    )

    print()
    print(
        f"Step 9: Building top "
        f"{len(top_evidence_results)} evidence items..."
    )

    evidence_builder = EvidenceBuilder(
        ocr_metadata_path=(
            index_dir / "ocr.json"
        ),
        visual_metadata_path=(
            index_dir / "visual_metadata.json"
        ),
    )

    evidence = evidence_builder.build(
        top_evidence_results
    )

    # --------------------------------------------------
    # 10. Display final evidence
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL EVIDENCE")
    print("=" * 80)

    for rank, item in enumerate(
        evidence,
        start=1,
    ):
        print()
        print(f"Evidence Rank: {rank}")
        print(f"Tile ID: {item.tile_id}")
        print(f"Page: {item.page_number}")
        print(
            f"RRF Score: "
            f"{item.rrf_score:.6f}"
        )
        print(
            f"BM25 Rank: "
            f"{item.bm25_rank}"
        )
        print(
            f"Visual Rank: "
            f"{item.visual_rank}"
        )
        print(
            f"Image Exists: "
            f"{item.image_path.exists()}"
        )

        preview = (
            item.text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 200:
            preview = preview[:200] + "..."

        print(
            f"OCR Preview: {preview}"
        )

    # --------------------------------------------------
    # 11. Validation
    # --------------------------------------------------

    print()
    print("Step 10: Validating pipeline...")

    if len(evidence) != evidence_top_k:
        raise AssertionError(
            "Unexpected number of evidence items."
        )

    for item in evidence:

        if not item.image_path.exists():
            raise AssertionError(
                f"Evidence image missing: "
                f"{item.image_path}"
            )

        if not item.text.strip():
            raise AssertionError(
                f"OCR text is empty for "
                f"{item.tile_id}"
            )

        if item.rrf_score <= 0:
            raise AssertionError(
                f"Invalid RRF score for "
                f"{item.tile_id}"
            )

    if evidence[0].tile_id != (
        "shared_sample_document-p0004-t0002"
    ):
        raise AssertionError(
            "Unexpected top evidence tile."
        )

    print()
    print(
        "End-to-end retrieval test successful!"
    )
    print()
    print(
        "Query -> BM25 -> SigLIP/FAISS -> "
        "RRF -> EvidenceBuilder"
    )
    print(
        "All components communicated successfully."
    )
    print()


if __name__ == "__main__":
    main()