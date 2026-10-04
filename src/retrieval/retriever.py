import json
from pathlib import Path

import faiss
import numpy as np

from src.embeddings.embedder import Embedder


INDEX_PATH = Path(
    "data/indexes/zx_bank.index"
)

METADATA_PATH = Path(
    "data/indexes/zx_bank_metadata.json"
)


class Retriever:

    def __init__(
        self,
        top_k: int = 5
    ):

        self.top_k = top_k

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        print(
            f"FAISS vectors loaded: "
            f"{self.index.ntotal}"
        )

        print("Loading metadata...")

        with METADATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        print(
            f"Metadata records loaded: "
            f"{len(self.metadata)}"
        )

        self.embedder = Embedder()

    def search(
        self,
        query: str
    ) -> list[dict]:

        if not query.strip():
            return []

        query_vector = self.embedder.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_vector = np.asarray(
            query_vector,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            self.top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            result = {
                **self.metadata[index],
                "score": float(score),
            }

            results.append(result)

        return results