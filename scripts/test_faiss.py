import numpy as np
import faiss


def main():
    # Create 3 fake 768-dimensional vectors.
    vectors = np.array(
        [
            np.ones(768),
            np.zeros(768),
            np.full(768, 0.5),
        ],
        dtype="float32",
    )

    # Create a FAISS index using inner product similarity.
    index = faiss.IndexFlatIP(768)

    print()
    print("FAISS index created!")
    print(f"Dimensions: {index.d}")
    print(f"Vectors before adding: {index.ntotal}")

    # Add our vectors.
    index.add(vectors)

    print(f"Vectors after adding: {index.ntotal}")

    # Use the first vector as the query.
    query = np.array(
        [np.ones(768)],
        dtype="float32",
    )

    # Search for the 3 nearest vectors.
    scores, indices = index.search(
        query,
        k=3,
    )

    print()
    print("FAISS search successful!")
    print()
    print("Rank | Vector ID | Score")

    for rank, (index_id, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        print(
            f"{rank:4} | "
            f"{index_id:9} | "
            f"{score:.4f}"
        )


if __name__ == "__main__":
    main()