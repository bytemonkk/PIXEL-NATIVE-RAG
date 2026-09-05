import json
from pathlib import Path

from app.retrieval.faiss_index import (
    FAISSVisualIndex,
)
from app.retrieval.visual_encoder import (
    SigLIPEncoder,
)


def load_metadata(
    metadata_path: Path,
) -> list[dict]:

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def search_query(
    query: str,
    encoder: SigLIPEncoder,
    index: FAISSVisualIndex,
    metadata: list[dict],
    top_k: int = 5,
) -> None:

    print()
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    query_vector = encoder.encode_text(
        text=query
    )

    results = index.search(
        query_vector=query_vector,
        top_k=top_k,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        record = metadata[
            result.vector_id
        ]

        print()
        print(
            f"Rank {rank}"
        )
        print(
            f"Score: {result.score:.6f}"
        )
        print(
            f"Vector ID: {result.vector_id}"
        )
        print(
            f"Tile ID: {record['tile_id']}"
        )
        print(
            f"Page: {record['page_number']}"
        )
        print(
            f"Image: {record['image_path']}"
        )


def main():

    index_dir = Path(
        "data/index/visual_index_sample"
    )

    faiss_path = (
        index_dir / "visual.index"
    )

    metadata_path = (
        index_dir / "visual_metadata.json"
    )

    print()
    print("Step 1: Loading metadata...")

    metadata = load_metadata(
        metadata_path
    )

    print(
        f"Metadata records: {len(metadata)}"
    )

    print()
    print("Step 2: Loading FAISS index...")

    index = FAISSVisualIndex.load(
        faiss_path
    )

    print(
        f"FAISS vectors: {index.size}"
    )

    print()
    print("Step 3: Loading SigLIP...")

    encoder = SigLIPEncoder()

    queries = [
        "architecture diagram",
        "ultrasound segmentation",
        "concept-aware fusion",
        "experimental results",
    ]

    print()
    print("Step 4: Running text-to-image retrieval...")

    for query in queries:

        search_query(
            query=query,
            encoder=encoder,
            index=index,
            metadata=metadata,
            top_k=5,
        )

    print()
    print("Visual text retrieval test complete!")


if __name__ == "__main__":
    main()