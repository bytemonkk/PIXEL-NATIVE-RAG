from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from scripts.search_bm25 import find_latest_ocr_file


def main():

    index_root = Path("data/index")

    ocr_path = find_latest_ocr_file(
        index_root
    )

    index_dir = ocr_path.parent

    print()
    print("Loading persisted BM25 index...")
    print(f"Index directory: {index_dir}")

    retriever = BM25Retriever.load(
        input_dir=index_dir
    )

    print(
        f"Documents loaded: "
        f"{len(retriever.documents)}"
    )

    query = "geometry guided"

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print()
    print("Persisted BM25 search successful!")
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 70)
        print(f"Rank: {rank}")
        print(f"Tile: {result.tile_id}")
        print(f"Page: {result.page_number}")
        print(f"Score: {result.score:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()