import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class BM25Document:
    tile_id: str
    page_number: int
    text: str


@dataclass
class BM25Result:
    tile_id: str
    page_number: int
    score: float
    text: str


class BM25Retriever:
    def __init__(
        self,
        documents: list[BM25Document],
    ):
        if not documents:
            raise ValueError(
                "BM25Retriever requires at least one document."
            )

        self.documents = documents

        self.tokenized_documents = [
            self._tokenize(document.text)
            for document in documents
        ]

        self.index = BM25Okapi(
            self.tokenized_documents
        )

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()

        tokens = re.findall(
            r"\b\w+\b",
            text,
        )

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25Result]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_tokens = self._tokenize(query)

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            document = self.documents[index]

            results.append(
                BM25Result(
                    tile_id=document.tile_id,
                    page_number=document.page_number,
                    score=float(scores[index]),
                    text=document.text,
                )
            )

        return results