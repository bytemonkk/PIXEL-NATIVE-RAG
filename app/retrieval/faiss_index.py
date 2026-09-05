from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class FAISSResult:
    vector_id: int
    score: float


class FAISSVisualIndex:
    def __init__(
        self,
        dimension: int,
    ):
        if dimension <= 0:
            raise ValueError(
                "Dimension must be greater than zero."
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(
        self,
        vectors: list[list[float]],
    ) -> None:

        if not vectors:
            raise ValueError(
                "Cannot add an empty vector list."
            )

        array = np.asarray(
            vectors,
            dtype="float32",
        )

        if array.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D array."
            )

        if array.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with "
                f"{self.dimension} dimensions, "
                f"got {array.shape[1]}."
            )

        self.index.add(array)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[FAISSResult]:

        if not query_vector:
            raise ValueError(
                "Query vector cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query = np.asarray(
            [query_vector],
            dtype="float32",
        )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query vector with "
                f"{self.dimension} dimensions, "
                f"got {query.shape[1]}."
            )

        scores, indices = self.index.search(
            query,
            min(top_k, self.size),
        )

        results = []

        for vector_id, score in zip(
            indices[0],
            scores[0],
        ):
            if vector_id == -1:
                continue

            results.append(
                FAISSResult(
                    vector_id=int(vector_id),
                    score=float(score),
                )
            )

        return results

    def save(
        self,
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(output_path),
        )

    @classmethod
    def load(
        cls,
        input_path: Path,
    ) -> "FAISSVisualIndex":

        if not input_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{input_path}"
            )

        index = faiss.read_index(
            str(input_path)
        )

        instance = cls.__new__(cls)

        instance.dimension = index.d
        instance.index = index

        return instance 