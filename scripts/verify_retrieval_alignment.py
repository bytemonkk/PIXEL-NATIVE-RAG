import json
from pathlib import Path


def main():
    index_dir = Path(
        "data/index/shared_sample_document"
    )

    bm25_metadata_path = (
        index_dir / "documents.json"
    )

    visual_metadata_path = (
        index_dir / "visual_metadata.json"
    )

    print()
    print("Loading BM25 metadata...")
    with bm25_metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        bm25_documents = json.load(file)

    print(
        f"BM25 records: {len(bm25_documents)}"
    )

    print()
    print("Loading visual metadata...")
    with visual_metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        visual_documents = json.load(file)

    print(
        f"Visual records: {len(visual_documents)}"
    )

    bm25_tile_ids = {
        document["tile_id"]
        for document in bm25_documents
    }

    visual_tile_ids = {
        document["tile_id"]
        for document in visual_documents
    }

    print()
    print("Checking tile ID alignment...")

    only_in_bm25 = (
        bm25_tile_ids - visual_tile_ids
    )

    only_in_visual = (
        visual_tile_ids - bm25_tile_ids
    )

    shared_tile_ids = (
        bm25_tile_ids & visual_tile_ids
    )

    print(
        f"Shared tile IDs: {len(shared_tile_ids)}"
    )
    print(
        f"Only in BM25: {len(only_in_bm25)}"
    )
    print(
        f"Only in Visual: {len(only_in_visual)}"
    )

    if only_in_bm25:
        print()
        print("Tiles only in BM25:")
        for tile_id in sorted(only_in_bm25):
            print(tile_id)

    if only_in_visual:
        print()
        print("Tiles only in Visual:")
        for tile_id in sorted(only_in_visual):
            print(tile_id)

    if (
        len(bm25_tile_ids) != len(bm25_documents)
    ):
        raise ValueError(
            "BM25 metadata contains duplicate tile IDs."
        )

    if (
        len(visual_tile_ids)
        != len(visual_documents)
    ):
        raise ValueError(
            "Visual metadata contains duplicate tile IDs."
        )

    if bm25_tile_ids != visual_tile_ids:
        raise ValueError(
            "BM25 and visual tile IDs are not aligned."
        )

    print()
    print(
        "BM25 ↔ Visual alignment verified!"
    )
    print(
        "Every BM25 tile has a corresponding "
        "visual tile."
    )
    print()


if __name__ == "__main__":
    main()