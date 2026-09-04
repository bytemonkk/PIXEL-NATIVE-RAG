from pathlib import Path
from uuid import uuid4

from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.tiler import ImageTiler


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

    total_tiles = 0

    for page in pages:
        tiles = tiler.tile_page(
            image_path=page.image_path,
            page_number=page.page_number,
            doc_id=doc_id,
        )

        total_tiles += len(tiles)

        print(
            f"Page {page.page_number}: "
            f"{len(tiles)} tiles"
        )

    print()
    print("Tiling successful!")
    print(f"Document ID: {doc_id}")
    print(f"Total tiles: {total_tiles}")


if __name__ == "__main__":
    main()