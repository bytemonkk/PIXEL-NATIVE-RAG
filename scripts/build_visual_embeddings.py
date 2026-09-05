import json
from pathlib import Path

from app.ingestion.tiler import ImageTiler
from app.ingestion.tile_quality import TileQualityFilter
from app.ingestion.pdf_renderer import PDFRenderer
from app.retrieval.visual_encoder import SigLIPEncoder


def main():
    pdf_path = Path("data/raw/sample.pdf")

    rendered_dir = Path("data/rendered")
    tiles_dir = Path("data/tiles")
    index_dir = Path("data/index")

    # Use the same deterministic document ID for this build.
    doc_id = "visual_index_sample"

    print()
    print("Step 1: Rendering PDF...")

    renderer = PDFRenderer(
        output_dir=rendered_dir,
        dpi=150,
    )

    pages = renderer.render(
        pdf_path=pdf_path,
        doc_id=doc_id,
    )

    print(f"Pages rendered: {len(pages)}")

    print()
    print("Step 2: Creating tiles...")

    tiler = ImageTiler(
        output_dir=tiles_dir,
        tile_width=1024,
        tile_height=1024,
        overlap=128,
    )

    all_tiles = []

    for page in pages:
        page_tiles = tiler.tile_page(
            image_path=page.image_path,
            page_number=page.page_number,
            doc_id=doc_id,
        )

        all_tiles.extend(page_tiles)

    print(f"Total tiles: {len(all_tiles)}")

    print()
    print("Step 3: Filtering tiles...")

    quality_filter = TileQualityFilter()

    quality_results = quality_filter.filter(
        all_tiles
    )

    kept_tile_ids = {
        result.tile_id
        for result in quality_results
        if result.keep
    }

    kept_tiles = [
        tile
        for tile in all_tiles
        if tile.tile_id in kept_tile_ids
    ]

    print(f"Kept tiles: {len(kept_tiles)}")
    print(
        f"Rejected tiles: "
        f"{len(all_tiles) - len(kept_tiles)}"
    )

    print()
    print("Step 4: Loading SigLIP...")

    encoder = SigLIPEncoder()

    print()
    print("Step 5: Encoding retained tiles...")

    embeddings = []
    metadata = []

    for position, tile in enumerate(
        kept_tiles,
        start=1,
    ):
        print(
            f"[{position}/{len(kept_tiles)}] "
            f"{tile.tile_id}"
        )

        vector = encoder.encode_image(
            image_path=tile.image_path
        )

        embeddings.append(vector)

        metadata.append(
            {
                "vector_id": position - 1,
                "tile_id": tile.tile_id,
                "page_number": tile.page_number,
                "x0": tile.x0,
                "y0": tile.y0,
                "x1": tile.x1,
                "y1": tile.y1,
                "image_path": str(tile.image_path),
            }
        )

    output_dir = index_dir / doc_id
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        output_dir / "visual_embeddings.json"
    )

    metadata_path = (
        output_dir / "visual_metadata.json"
    )

    with embeddings_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            embeddings,
            file,
        )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("Visual embedding generation successful!")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Vectors generated: {len(embeddings)}")
    print(
        f"Vector dimensions: "
        f"{len(embeddings[0])}"
    )


if __name__ == "__main__":
    main()