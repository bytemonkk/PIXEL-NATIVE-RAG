from pathlib import Path

from app.retrieval.evidence_builder import (
    EvidenceBuilder,
)
from app.retrieval.rrf import (
    RRFResult,
)


def main():

    index_dir = Path(
        "data/index/shared_sample_document"
    )

    builder = EvidenceBuilder(
        ocr_metadata_path=(
            index_dir / "ocr.json"
        ),
        visual_metadata_path=(
            index_dir / "visual_metadata.json"
        ),
    )

    rrf_results = [
        RRFResult(
            tile_id=(
                "shared_sample_document-p0004-t0002"
            ),
            score=0.032787,
            ranks={
                "bm25": 1,
                "visual": 1,
            },
        )
    ]

    evidence = builder.build(
        rrf_results
    )

    print()
    print(
        "Evidence builder test results:"
    )
    print()

    print(
        f"Evidence items: "
        f"{len(evidence)}"
    )

    item = evidence[0]

    print(
        f"Tile ID: {item.tile_id}"
    )

    print(
        f"Page: {item.page_number}"
    )

    print(
        f"Image: {item.image_path}"
    )

    print(
        f"RRF score: "
        f"{item.rrf_score:.6f}"
    )

    print(
        f"BM25 rank: "
        f"{item.bm25_rank}"
    )

    print(
        f"Visual rank: "
        f"{item.visual_rank}"
    )

    print(
        f"Image exists: "
        f"{item.image_path.exists()}"
    )

    print(
        f"OCR characters: "
        f"{len(item.text)}"
    )

    print()

    if len(evidence) != 1:
        raise AssertionError(
            "Expected one evidence item."
        )

    if item.tile_id != (
        "shared_sample_document-p0004-t0002"
    ):
        raise AssertionError(
            "Incorrect tile ID."
        )

    if item.page_number != 4:
        raise AssertionError(
            "Incorrect page number."
        )

    if item.bm25_rank != 1:
        raise AssertionError(
            "Incorrect BM25 rank."
        )

    if item.visual_rank != 1:
        raise AssertionError(
            "Incorrect visual rank."
        )

    if not item.image_path.exists():
        raise AssertionError(
            "Evidence image does not exist."
        )

    if not item.text.strip():
        raise AssertionError(
            "OCR text is empty."
        )

    print(
        "Evidence builder test successful!"
    )


if __name__ == "__main__":
    main()