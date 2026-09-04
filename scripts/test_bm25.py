from app.retrieval.bm25 import (
    BM25Document,
    BM25Retriever,
)


def main():

    documents = [
        BM25Document(
            tile_id="tile-001",
            page_number=1,
            text=(
                "Deep learning is used for "
                "medical image segmentation."
            ),
        ),
        BM25Document(
            tile_id="tile-002",
            page_number=2,
            text=(
                "Geometry guided vision text "
                "fusion improves ultrasound segmentation."
            ),
        ),
        BM25Document(
            tile_id="tile-003",
            page_number=3,
            text=(
                "Experimental results compare "
                "different segmentation methods."
            ),
        ),
    ]

    retriever = BM25Retriever(
        documents=documents
    )

    query = "ultrasound segmentation"

    results = retriever.search(
        query=query,
        top_k=3,
    )

    print()
    print("BM25 retrieval successful!")
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank {rank}"
        )
        print(
            f"Tile: {result.tile_id}"
        )
        print(
            f"Page: {result.page_number}"
        )
        print(
            f"Score: {result.score:.4f}"
        )
        print(
            f"Text: {result.text}"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()