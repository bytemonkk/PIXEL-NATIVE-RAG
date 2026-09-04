from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    tile_id: str
    page_number: int
    image_path: str
    text: str
    confidence: float


class TileOCR:
    def __init__(
        self,
        language: str = "eng",
        config: str = "--psm 6",
    ):
        self.language = language
        self.config = config

    def extract(
        self,
        image_path: Path,
        tile_id: str,
        page_number: int,
    ) -> OCRResult:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Tile image not found: {image_path}"
            )

        with Image.open(image_path) as image:

            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.config,
            )

            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )

        confidences = []

        for confidence in data["conf"]:
            try:
                value = float(confidence)
            except ValueError:
                continue

            if value >= 0:
                confidences.append(value)

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        return OCRResult(
            tile_id=tile_id,
            page_number=page_number,
            image_path=str(image_path),
            text=text.strip(),
            confidence=average_confidence,
        )


def save_ocr_results(
    results: list[OCRResult],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = [
        asdict(result)
        for result in results
    ]

    temporary_path = output_path.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(output_path)