from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat


@dataclass
class TileQualityResult:
    tile_id: str
    image_path: Path

    is_blank: bool
    is_duplicate: bool
    keep: bool

    brightness: float
    duplicate_of: str | None = None


class TileQualityFilter:
    def __init__(
        self,
        blank_brightness_threshold: float = 245.0,
        duplicate_difference_threshold: float = 2.0,
    ):
        self.blank_brightness_threshold = (
            blank_brightness_threshold
        )

        self.duplicate_difference_threshold = (
            duplicate_difference_threshold
        )

    def _brightness(self, image: Image.Image) -> float:
        grayscale = image.convert("L")

        statistics = ImageStat.Stat(grayscale)

        return statistics.mean[0]

    def _difference(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
    ) -> float:
        image_a = image_a.convert("L").resize((64, 64))
        image_b = image_b.convert("L").resize((64, 64))

        pixels_a = list(image_a.getdata())
        pixels_b = list(image_b.getdata())

        differences = [
            abs(a - b)
            for a, b in zip(pixels_a, pixels_b)
        ]

        return sum(differences) / len(differences)

    def filter(
        self,
        tiles: list,
    ) -> list[TileQualityResult]:

        results: list[TileQualityResult] = []

        kept_images: list[
            tuple[str, Image.Image]
        ] = []

        for tile in tiles:

            image = Image.open(tile.image_path)

            brightness = self._brightness(image)

            is_blank = (
                brightness
                >= self.blank_brightness_threshold
            )

            is_duplicate = False
            duplicate_of = None

            if not is_blank:

                for (
                    previous_tile_id,
                    previous_image,
                ) in kept_images:

                    difference = self._difference(
                        image,
                        previous_image,
                    )

                    if (
                        difference
                        <= self.duplicate_difference_threshold
                    ):
                        is_duplicate = True
                        duplicate_of = (
                            previous_tile_id
                        )
                        break

            keep = not is_blank and not is_duplicate

            result = TileQualityResult(
                tile_id=tile.tile_id,
                image_path=tile.image_path,
                is_blank=is_blank,
                is_duplicate=is_duplicate,
                keep=keep,
                brightness=brightness,
                duplicate_of=duplicate_of,
            )

            results.append(result)

            if keep:
                kept_images.append(
                    (
                        tile.tile_id,
                        image.copy(),
                    )
                )

            image.close()

        for _, image in kept_images:
            image.close()

        return results