import json
from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.rrf import RRFFusion
from app.retrieval.visual_encoder import SigLIPEncoder
from app.retrieval.faiss_index import FAISSVisualIndex


def main():
    index_dir = Path(
        "data/index/shared_sample_document"
    )

    query = "experimental results"
    top_k = 5

    # --------------------------------------------------
    # 1. Load BM25
    # --------------------------------------------------

    print()
    print("Step 1: Loading BM25...")

    bm25 = BM25Retriever.load(
        index_dir
    )

    print(
        f"BM25 documents: "
        f"{len(bm25.documents)}"
    )

    # --------------------------------------------------
    # 2. Run BM25
    # --------------------------------------------------

    print()
    print("Step 2: Running BM25...")
    print(f"Query: {query}")

    bm25_results = bm25.search(
        query=query,
        top_k=top_k,
    )

    print()
    print("BM25 ranking:")

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
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
    print("Step 4: Loading FAISS...")

    faiss_index = FAISSVisualIndex.load(
        index_dir / "visual.index"
    )

    print(
        f"FAISS vectors: "
        f"{faiss_index.size}"
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
    # 7. Run visual retrieval
    # --------------------------------------------------

    print()
    print("Step 7: Running visual retrieval...")

    visual_results = faiss_index.search(
        query_vector=query_vector,
        top_k=top_k,
    )

    print()
    print("Visual ranking:")

    visual_tile_ids = []

    for rank, result in enumerate(
        visual_results,
        start=1,
    ):
        metadata = metadata_by_vector_id[
            result.vector_id
        ]

        tile_id = metadata["tile_id"]

        visual_tile_ids.append(tile_id)

        print(
            f"Rank {rank}: "
            f"{tile_id} "
            f"score={result.score:.6f}"
        )

    # --------------------------------------------------
    # 8. Extract BM25 tile IDs
    # --------------------------------------------------

    bm25_tile_ids = [
        result.tile_id
        for result in bm25_results
    ]

    # --------------------------------------------------
    # 9. RRF fusion
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

    # --------------------------------------------------
    # 10. Display final ranking
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("HYBRID RRF RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        fused_results,
        start=1,
    ):
        print()
        print(
            f"Rank {rank}: "
            f"{result.tile_id}"
        )

        print(
            f"RRF score: "
            f"{result.score:.6f}"
        )

        print(
            f"Retriever ranks: "
            f"{result.ranks}"
        )

    print()
    print(
        "Hybrid retrieval test complete!"
    )


if __name__ == "__main__":
    main()