from dataclasses import dataclass
from pathlib import Path

from PIL import Image

##Those coordinates are extremely important later.
##Document → abc
##Page → 2
##Tile → 3
##Coordinates → x0,y0,x1,y1
##That's how we'll eventually be able to trace an answer back to the exact visual evidence.

@dataclass
class Tile:
    tile_id: str
    page_number: int
    sequence: int

    x0: int
    y0: int
    x1: int
    y1: int

    image_path: Path


class ImageTiler:
    def __init__(
        self,
        output_dir: Path,
        tile_width: int = 1024,
        tile_height: int = 1024,
        overlap: int = 128,
    ):
        if overlap >= tile_width:
            raise ValueError(
                "Overlap must be smaller than tile width."
            )

        if overlap >= tile_height:
            raise ValueError(
                "Overlap must be smaller than tile height."
            )

        self.output_dir = output_dir
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap = overlap

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def tile_page(
        self,
        image_path: Path,
        page_number: int,
        doc_id: str,
    ) -> list[Tile]:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path)

        image_width, image_height = image.size

        step_x = self.tile_width - self.overlap
        step_y = self.tile_height - self.overlap

        page_output_dir = (
            self.output_dir
            / doc_id
            / f"page_{page_number:04d}"
        )

        page_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tiles: list[Tile] = []

        sequence = 0

        for y0 in range(0, image_height, step_y):
            for x0 in range(0, image_width, step_x):

                x1 = min(
                    x0 + self.tile_width,
                    image_width,
                )

                y1 = min(
                    y0 + self.tile_height,
                    image_height,
                )

                crop = image.crop(
                    (x0, y0, x1, y1)
                )

                tile_id = (
                    f"{doc_id}"
                    f"-p{page_number:04d}"
                    f"-t{sequence:04d}"
                )

                image_output_path = (
                    page_output_dir
                    / f"{tile_id}.png"
                )

                crop.save(
                    image_output_path,
                    format="PNG",
                )

                tiles.append(
                    Tile(
                        tile_id=tile_id,
                        page_number=page_number,
                        sequence=sequence,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        image_path=image_output_path,
                    )
                )

                sequence += 1

        image.close()

        return tiles