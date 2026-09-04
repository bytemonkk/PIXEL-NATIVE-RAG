from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image


@dataclass
class OCRResult:
    tile_id: str
    image_path: Path
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
            image_path=image_path,
            text=text.strip(),
            confidence=average_confidence,
        )