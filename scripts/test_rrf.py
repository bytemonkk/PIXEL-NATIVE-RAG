from app.retrieval.rrf import RRFFusion


def main():
    fusion = RRFFusion(k=60)

    ranked_lists = {
        "bm25": [
            "tile-A",
            "tile-B",
            "tile-C",
        ],
        "visual": [
            "tile-C",
            "tile-A",
            "tile-D",
        ],
    }

    results = fusion.fuse(
        ranked_lists
    )

    print()
    print("RRF test results:")
    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
            f"{result.tile_id}"
        )

        print(
            f"Score: "
            f"{result.score:.6f}"
        )

        print(
            f"Ranks: "
            f"{result.ranks}"
        )

        print()

    if results[0].tile_id != "tile-A":
        raise AssertionError(
            "Expected tile-A to rank first."
        )

    if results[1].tile_id != "tile-C":
        raise AssertionError(
            "Expected tile-C to rank second."
        )

    print(
        "RRF test successful!"
    )


if __name__ == "__main__":
    main()