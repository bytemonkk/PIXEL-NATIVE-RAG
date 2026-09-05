import json
from pathlib import Path

from app.retrieval.faiss_index import (
    FAISSVisualIndex,
)


def main():

    index_dir = Path(
        "data/index/shared_sample_document"
    )

    embeddings_path = (
        index_dir / "visual_embeddings.json"
    )

    metadata_path = (
        index_dir / "visual_metadata.json"
    )

    faiss_path = (
        index_dir / "visual.index"
    )

    print()
    print("Step 1: Loading visual embeddings...")

    with embeddings_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        embeddings = json.load(file)

    print(
        f"Embeddings loaded: {len(embeddings)}"
    )

    if not embeddings:
        raise ValueError(
            "No visual embeddings found."
        )

    dimension = len(embeddings[0])

    for position, vector in enumerate(
        embeddings
    ):
        if len(vector) != dimension:
            raise ValueError(
                f"Embedding {position} has "
                f"{len(vector)} dimensions; "
                f"expected {dimension}."
            )

    print(
        f"Embedding dimensions: {dimension}"
    )

    print()
    print("Step 2: Loading tile metadata...")

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    print(
        f"Metadata records: {len(metadata)}"
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding count does not match "
            "metadata count."
        )

    for expected_id, record in enumerate(
        metadata
    ):
        actual_id = record["vector_id"]

        if actual_id != expected_id:
            raise ValueError(
                f"Metadata vector ID mismatch: "
                f"expected {expected_id}, "
                f"got {actual_id}."
            )

    print(
        "Embedding ↔ metadata mapping verified."
    )

    print()
    print("Step 3: Building FAISS index...")

    index = FAISSVisualIndex(
        dimension=dimension
    )

    index.add(embeddings)

    print(
        f"FAISS vectors: {index.size}"
    )

    if index.size != len(embeddings):
        raise RuntimeError(
            "FAISS index size does not match "
            "embedding count."
        )

    print()
    print("Step 4: Saving FAISS index...")

    index.save(faiss_path)

    print(
        f"FAISS index saved: {faiss_path}"
    )

    print()
    print("Step 5: Reloading FAISS index...")

    loaded_index = (
        FAISSVisualIndex.load(
            faiss_path
        )
    )

    print(
        f"Reloaded vectors: "
        f"{loaded_index.size}"
    )

    print()
    print("Step 6: Testing visual search...")

    query_vector = embeddings[0]

    results = loaded_index.search(
        query_vector=query_vector,
        top_k=5,
    )

    print()
    print("Top visual results:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        record = metadata[
            result.vector_id
        ]

        print(
            f"Rank {rank}: "
            f"vector_id={result.vector_id}, "
            f"score={result.score:.6f}, "
            f"tile={record['tile_id']}, "
            f"page={record['page_number']}"
        )

    print()
    print("FAISS visual index build successful!")


if __name__ == "__main__":
    main()