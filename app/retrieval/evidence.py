from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvidenceItem:
    tile_id: str
    page_number: int

    image_path: Path

    text: str

    rrf_score: float

    bm25_rank: int | None
    visual_rank: int | None

    x0: int
    y0: int
    x1: int
    y1: int