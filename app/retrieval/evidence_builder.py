import json
from pathlib import Path

from app.retrieval.evidence import EvidenceItem
from app.retrieval.rrf import RRFResult


class EvidenceBuilder:
    def __init__(
        self,
        ocr_metadata_path: Path,
        visual_metadata_path: Path,
    ):
        self.ocr_metadata_path = ocr_metadata_path
        self.visual_metadata_path = visual_metadata_path

        self.ocr_by_tile_id = (
            self._load_ocr_metadata()
        )

        self.visual_by_tile_id = (
            self._load_visual_metadata()
        )

    def _load_ocr_metadata(self) -> dict:
        with self.ocr_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        return {
            record["tile_id"]: record
            for record in records
        }

    def _load_visual_metadata(self) -> dict:
        with self.visual_metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        return {
            record["tile_id"]: record
            for record in records
        }

    def build(
        self,
        rrf_results: list[RRFResult],
    ) -> list[EvidenceItem]:

        evidence_items = []

        for result in rrf_results:

            tile_id = result.tile_id

            if tile_id not in self.ocr_by_tile_id:
                raise KeyError(
                    f"OCR metadata not found for "
                    f"tile: {tile_id}"
                )

            if tile_id not in self.visual_by_tile_id:
                raise KeyError(
                    f"Visual metadata not found for "
                    f"tile: {tile_id}"
                )

            ocr = self.ocr_by_tile_id[
                tile_id
            ]

            visual = self.visual_by_tile_id[
                tile_id
            ]

            evidence_items.append(
                EvidenceItem(
                    tile_id=tile_id,
                    page_number=ocr["page_number"],
                    image_path=Path(
                        visual["image_path"]
                    ),
                    text=ocr["text"],
                    rrf_score=result.score,
                    bm25_rank=result.ranks.get(
                        "bm25"
                    ),
                    visual_rank=result.ranks.get(
                        "visual"
                    ),
                    x0=visual["x0"],
                    y0=visual["y0"],
                    x1=visual["x1"],
                    y1=visual["y1"],
                )
            )

        return evidence_items