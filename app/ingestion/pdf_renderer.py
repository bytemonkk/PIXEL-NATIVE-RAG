from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class RenderedPage:
    doc_id: str
    source: str
    page_number: int
    image_path: Path
    width: int
    height: int


class PDFRenderer:
    def __init__(
        self,
        output_dir: Path,
        dpi: int = 150,
    ):
        self.output_dir = output_dir
        self.dpi = dpi

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def render(
        self,
        pdf_path: Path,
        doc_id: str,
    ) -> list[RenderedPage]:

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        document = pymupdf.open(pdf_path)

        pages: list[RenderedPage] = []

        zoom = self.dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        document_dir = self.output_dir / doc_id

        document_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            for page_number, page in enumerate(
                document,
                start=1,
            ):
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image_path = (
                    document_dir
                    / f"page_{page_number:04d}.png"
                )

                pixmap.save(str(image_path))

                pages.append(
                    RenderedPage(
                        doc_id=doc_id,
                        source=str(pdf_path),
                        page_number=page_number,
                        image_path=image_path,
                        width=pixmap.width,
                        height=pixmap.height,
                    )
                )

        finally:
            document.close()

        return pages