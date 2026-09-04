import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

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

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        text = text.lower()

        return re.findall(
            r"\b\w+\b",
            text,
        )

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

    def save(
        self,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            output_dir / "bm25.pkl"
        )

        metadata_path = (
            output_dir / "documents.json"
        )

        with index_path.open("wb") as file:
            pickle.dump(
                self.index,
                file,
            )

        import json

        metadata = [
            asdict(document)
            for document in self.documents
        ]

        temporary_path = metadata_path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(
            metadata_path
        )

    @classmethod
    def load(
        cls,
        input_dir: Path,
    ) -> "BM25Retriever":

        index_path = (
            input_dir / "bm25.pkl"
        )

        metadata_path = (
            input_dir / "documents.json"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"BM25 metadata not found: {metadata_path}"
            )

        import json

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_documents = json.load(file)

        documents = [
            BM25Document(**document)
            for document in raw_documents
        ]

        retriever = cls.__new__(cls)

        retriever.documents = documents

        retriever.tokenized_documents = [
            retriever._tokenize(document.text)
            for document in documents
        ]

        with index_path.open("rb") as file:
            retriever.index = pickle.load(file)

        return retriever