from pathlib import Path
# from uuid import uuid4

from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.tiler import ImageTiler
from app.ingestion.tile_quality import TileQualityFilter
from app.ingestion.ocr import (TileOCR, save_ocr_results)


def main():
    pdf_path = Path("data/raw/sample.pdf")

    # doc_id = uuid4().hex
    doc_id = "shared_sample_document"

    # 1. Render PDF
    renderer = PDFRenderer(
        output_dir=Path("data/rendered"),
        dpi=150,
    )

    pages = renderer.render(
        pdf_path=pdf_path,
        doc_id=doc_id,
    )

    # 2. Create tiles
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

    # 3. Filter tiles
    quality_filter = TileQualityFilter()

    quality_results = quality_filter.filter(
        all_tiles
    )

    kept_tiles = [
        tile
        for tile, quality in zip(
            all_tiles,
            quality_results,
        )
        if quality.keep
    ]

    # 4. OCR kept tiles
    ocr = TileOCR()

    ocr_results = []

    for tile in kept_tiles:
        result = ocr.extract(
            image_path=tile.image_path,
            tile_id=tile.tile_id,
            page_number=tile.page_number,
        )

        ocr_results.append(result)

    # 5. Save OCR metadata
    ocr_output_path = (
        Path("data/index")
        / doc_id
        / "ocr.json"
    )

    save_ocr_results(
        results=ocr_results,
        output_path=ocr_output_path,
    )

    # 5. Report
    print()
    print("OCR pipeline successful!")
    print(f"Document ID: {doc_id}")
    print(f"Total tiles: {len(all_tiles)}")
    print(f"Kept tiles: {len(kept_tiles)}")
    print(f"OCR results: {len(ocr_results)}")
    print(f"OCR metadata: {ocr_output_path}")
    print()

    for result in ocr_results:
        print("=" * 70)
        print(f"Tile: {result.tile_id}")
        print(f"Confidence: {result.confidence:.2f}")
        print()
        print(result.text[:1000])

    print("=" * 70)


if __name__ == "__main__":
    main()