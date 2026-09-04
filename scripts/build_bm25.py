from pathlib import Path

from app.retrieval.bm25 import (
    BM25Retriever,
)
from scripts.search_bm25 import (
    find_latest_ocr_file,
    load_ocr_documents,
)


def main():

    index_dir = Path("data/index")

    # Find the OCR artifact produced by Block 05.
    ocr_path = find_latest_ocr_file(
        index_dir
    )

    print()
    print("Building BM25 index...")
    print(f"OCR source: {ocr_path}")

    # Load OCR records.
    documents = load_ocr_documents(
        ocr_path
    )

    print(
        f"Documents loaded: {len(documents)}"
    )

    # Build BM25.
    retriever = BM25Retriever(
        documents=documents
    )

    # Store the index beside the OCR metadata.
    output_dir = ocr_path.parent

    retriever.save(
        output_dir=output_dir
    )

    print()
    print("BM25 index built successfully!")
    print(f"Index: {output_dir / 'bm25.pkl'}")
    print(
        f"Metadata: {output_dir / 'documents.json'}"
    )
    print()


if __name__ == "__main__":
    main()