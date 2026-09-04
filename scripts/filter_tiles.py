from pathlib import Path
from uuid import uuid4

from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.tiler import ImageTiler
from app.ingestion.tile_quality import TileQualityFilter


def main():
    pdf_path = Path("data/raw/sample.pdf")

    doc_id = uuid4().hex

    renderer = PDFRenderer(
        output_dir=Path("data/rendered"),
        dpi=150,
    )

    pages = renderer.render(
        pdf_path=pdf_path,
        doc_id=doc_id,
    )

    tiler = ImageTiler(
        output_dir=Path("data/tiles"),
        tile_width=1024,
        tile_height=1024,
        overlap=128,
    )

    all_tiles = []

    for page in pages:
        tiles = tiler.tile_page(
            image_path=page.image_path,
            page_number=page.page_number,
            doc_id=doc_id,
        )

        all_tiles.extend(tiles)

    quality_filter = TileQualityFilter()

    results = quality_filter.filter(
        all_tiles
    )

    kept = [
        result
        for result in results
        if result.keep
    ]

    blank = [
        result
        for result in results
        if result.is_blank
    ]

    duplicates = [
        result
        for result in results
        if result.is_duplicate
    ]

    print()
    print("Tile quality filtering successful!")
    print(f"Document ID: {doc_id}")
    print(f"Total tiles: {len(results)}")
    print(f"Blank tiles: {len(blank)}")
    print(f"Duplicate tiles: {len(duplicates)}")
    print(f"Kept tiles: {len(kept)}")
    print()

    for result in results:
        status = "KEEP"

        if result.is_blank:
            status = "BLANK"

        elif result.is_duplicate:
            status = (
                f"DUPLICATE "
                f"of {result.duplicate_of}"
            )

        print(
            f"{result.tile_id} | "
            f"{status} | "
            f"brightness={result.brightness:.2f}"
        )


if __name__ == "__main__":
    main()