from pathlib import Path

from app.retrieval.visual_encoder import (
    SigLIPEncoder,
)


def main():

    tile_path = Path(
        "data/tiles/"
        "300c7e564895423e980d478df58dc004/"
        "page_0002/"
        "300c7e564895423e980d478df58dc004"
        "-p0002-t0000.png"
    )

    encoder = SigLIPEncoder()

    vector = encoder.encode_image(
        image_path=tile_path
    )

    print()
    print("Visual embedding successful!")
    print(f"Image: {tile_path}")
    print(
        f"Embedding dimensions: {len(vector)}"
    )
    print(
        f"Vector type: {type(vector)}"
    )
    print(
        f"Element type: {type(vector[0])}"
    )
    print()
    print("First 10 values:")

    for value in vector[:10]:
        print(f"{value:.6f}")


if __name__ == "__main__":
    main()