from dataclasses import dataclass


@dataclass
class RRFResult:
    tile_id: str
    score: float
    ranks: dict[str, int]


class RRFFusion:
    def __init__(
        self,
        k: int = 60,
    ):
        if k <= 0:
            raise ValueError(
                "RRF k must be greater than zero."
            )

        self.k = k

    def fuse(
        self,
        ranked_lists: dict[
            str,
            list[str],
        ],
    ) -> list[RRFResult]:

        if not ranked_lists:
            raise ValueError(
                "At least one ranked list is required."
            )

        scores: dict[str, float] = {}

        ranks: dict[
            str,
            dict[str, int],
        ] = {}

        for retriever_name, tile_ids in ranked_lists.items():

            for rank, tile_id in enumerate(
                tile_ids,
                start=1,
            ):

                score = 1 / (
                    self.k + rank
                )

                scores[tile_id] = (
                    scores.get(tile_id, 0.0)
                    + score
                )

                if tile_id not in ranks:
                    ranks[tile_id] = {}

                ranks[tile_id][
                    retriever_name
                ] = rank

        ranked_tiles = sorted(
            scores,
            key=lambda tile_id: scores[tile_id],
            reverse=True,
        )

        return [
            RRFResult(
                tile_id=tile_id,
                score=scores[tile_id],
                ranks=ranks[tile_id],
            )
            for tile_id in ranked_tiles
        ]