from pathlib import Path

from app.retrieval.faiss_index import (
    FAISSVisualIndex,
)
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

    print()
    print("Step 1: Loading SigLIP...")

    encoder = SigLIPEncoder()

    print()
    print("Step 2: Encoding tile...")

    vector = encoder.encode_image(
        image_path=tile_path
    )

    print(
        f"Embedding dimensions: {len(vector)}"
    )

    print()
    print("Step 3: Creating FAISS index...")

    index = FAISSVisualIndex(
        dimension=len(vector)
    )

    print()
    print("Step 4: Adding tile embedding...")

    index.add([vector])

    print(
        f"FAISS index size: {index.size}"
    )

    print()
    print("Step 5: Searching using same tile...")

    results = index.search(
        query_vector=vector,
        top_k=1,
    )

    print()

    if not results:
        raise RuntimeError(
            "FAISS returned no results."
        )

    result = results[0]

    print("Visual retrieval successful!")
    print(f"Vector ID: {result.vector_id}")
    print(f"Similarity score: {result.score:.6f}")

    print()
    print("Expected:")
    print("Vector ID: 0")
    print("Similarity score: approximately 1.0")


if __name__ == "__main__":
    main()