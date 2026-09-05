from app.retrieval.hybrid_retriever import (
    HybridRetriever,
)


def main():
    retriever = HybridRetriever()

    print()
    print(
        "HybridRetriever created successfully!"
    )

    print(
        f"Class: {retriever.__class__.__name__}"
    )

    print()
    print(
        "HybridRetriever interface test successful!"
    )
    print()


if __name__ == "__main__":
    main()