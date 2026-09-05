from app.retrieval.visual_encoder import (
    SigLIPEncoder,
)


def main():

    query = "architecture diagram"

    print()
    print("Loading SigLIP...")

    encoder = SigLIPEncoder()

    print()
    print("Encoding text...")
    print(f"Query: {query}")

    vector = encoder.encode_text(
        text=query
    )

    print()
    print("Text embedding successful!")
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