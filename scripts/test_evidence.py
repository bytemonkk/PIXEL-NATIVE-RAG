from pathlib import Path

from app.retrieval.evidence import EvidenceItem


def main():
    evidence = EvidenceItem(
        tile_id="shared_sample_document-p0004-t0002",
        page_number=4,
        image_path=Path(
            "data/tiles/shared_sample_document/"
            "page_0004/"
            "shared_sample_document-p0004-t0002.png"
        ),
        text="Example OCR text.",
        rrf_score=0.032787,
        bm25_rank=1,
        visual_rank=1,
        x0=0,
        y0=896,
        x1=1024,
        y1=1650,
    )

    print()
    print("Evidence item created successfully!")
    print()

    print(f"Tile ID: {evidence.tile_id}")
    print(f"Page: {evidence.page_number}")
    print(f"Image: {evidence.image_path}")
    print(f"RRF score: {evidence.rrf_score:.6f}")
    print(f"BM25 rank: {evidence.bm25_rank}")
    print(f"Visual rank: {evidence.visual_rank}")
    print(
        "Image exists:",
        evidence.image_path.exists(),
    )

    print()

    if evidence.tile_id != (
        "shared_sample_document-p0004-t0002"
    ):
        raise AssertionError(
            "Tile ID mismatch."
        )

    if evidence.page_number != 4:
        raise AssertionError(
            "Page number mismatch."
        )

    if not evidence.image_path.exists():
        raise AssertionError(
            "Evidence image does not exist."
        )

    print("Evidence model test successful!")


if __name__ == "__main__":
    main()