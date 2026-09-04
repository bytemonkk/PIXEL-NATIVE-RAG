import json
from pathlib import Path

from app.retrieval.bm25 import (
    BM25Document,
    BM25Retriever,
)


def load_ocr_documents(
    ocr_path: Path,
) -> list[BM25Document]:

    if not ocr_path.exists():
        raise FileNotFoundError(
            f"OCR metadata not found: {ocr_path}"
        )

    with ocr_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    documents = []

    for record in records:
        text = record.get("text", "").strip()

        if not text:
            continue

        documents.append(
            BM25Document(
                tile_id=record["tile_id"],
                page_number=record["page_number"],
                text=text,
            )
        )

    if not documents:
        raise ValueError(
            "No OCR documents with text were found."
        )

    return documents


def find_latest_ocr_file(
    index_dir: Path,
) -> Path:

    ocr_files = list(
        index_dir.glob(
            "*/ocr.json"
        )
    )

    if not ocr_files:
        raise FileNotFoundError(
            f"No ocr.json files found in {index_dir}"
        )

    return max(
        ocr_files,
        key=lambda path: path.stat().st_mtime,
    )


def main():

    index_dir = Path("data/index")

    ocr_path = find_latest_ocr_file(
        index_dir
    )

    print()
    print("Loading OCR metadata...")
    print(f"Source: {ocr_path}")

    documents = load_ocr_documents(
        ocr_path
    )

    print(
        f"OCR documents loaded: {len(documents)}"
    )

    retriever = BM25Retriever(
        documents=documents
    )

    query = "geometry guided"

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print()
    print("BM25 retrieval successful!")
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
        print()
        print("OCR text:")
        print(result.text[:1000])

    print("=" * 70)


if __name__ == "__main__":
    main()