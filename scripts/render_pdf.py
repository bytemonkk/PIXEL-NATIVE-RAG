from pathlib import Path
from uuid import uuid4

from app.ingestion.pdf_renderer import PDFRenderer


def main():
    pdf_path = Path("data/raw/sample.pdf")

    renderer = PDFRenderer(
        output_dir=Path("data/rendered"),
        dpi=150,
    )

    doc_id = uuid4().hex

    pages = renderer.render(
        pdf_path=pdf_path,
        doc_id=doc_id,
    )

    print()
    print("PDF rendering successful!")
    print(f"Document ID: {doc_id}")
    print(f"Pages rendered: {len(pages)}")
    print()

    for page in pages:
        print(
            f"Page {page.page_number}: "
            f"{page.image_path} "
            f"({page.width}x{page.height})"
        )


if __name__ == "__main__":
    main()