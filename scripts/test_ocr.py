from pathlib import Path

from app.ingestion.ocr import TileOCR


def main():
    tile_id = (
        "300c7e564895423e980d478df58dc004"
        "-p0002-t0000"
    )

    image_path = Path(
        "data/tiles/"
        "300c7e564895423e980d478df58dc004/"
        "page_0002/"
        f"{tile_id}.png"
    )

    ocr = TileOCR()

    result = ocr.extract(
        image_path=image_path,
        tile_id=tile_id,
    )

    print()
    print("OCR successful!")
    print(f"Tile ID: {result.tile_id}")
    print(f"Confidence: {result.confidence:.2f}")
    print()
    print("----- OCR TEXT -----")
    print(result.text)
    print("--------------------")


if __name__ == "__main__":
    main()