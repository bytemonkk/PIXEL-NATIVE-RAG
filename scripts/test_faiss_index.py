from pathlib import Path

from app.retrieval.faiss_index import (
    FAISSVisualIndex,
)


def main():

    index = FAISSVisualIndex(
        dimension=768
    )

    vectors = [
        [1.0] * 768,
        [0.0] * 768,
        [0.5] * 768,
    ]

    index.add(vectors)

    print()
    print("FAISSVisualIndex working!")
    print(f"Dimension: {index.dimension}")
    print(f"Index size: {index.size}")

    query = [1.0] * 768

    results = index.search(
        query_vector=query,
        top_k=3,
    )

    print()
    print("Search results:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
            f"vector_id={result.vector_id}, "
            f"score={result.score:.4f}"
        )

    # Test persistence.
    index_path = Path(
        "data/index/test_faiss.index"
    )

    index.save(index_path)

    print()
    print(f"Index saved to: {index_path}")

    loaded_index = (
        FAISSVisualIndex.load(index_path)
    )

    print(
        f"Loaded index size: "
        f"{loaded_index.size}"
    )

    loaded_results = loaded_index.search(
        query_vector=query,
        top_k=3,
    )

    print()
    print("Loaded index search:")

    for rank, result in enumerate(
        loaded_results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
            f"vector_id={result.vector_id}, "
            f"score={result.score:.4f}"
        )


if __name__ == "__main__":
    main()