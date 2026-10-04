import json
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_PATH = Path(
    "data/processed/zx_bank_embeddings.json"
)

INDEX_PATH = Path(
    "data/indexes/zx_bank.index"
)

METADATA_PATH = Path(
    "data/indexes/zx_bank_metadata.json"
)


def load_embeddings():

    with EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    return records


def build_index(records):

    vectors = np.array(
        [
            record["embedding"]
            for record in records
        ],
        dtype="float32"
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(vectors)

    return index


def save_index(index):

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )


def save_metadata(records):

    metadata = []

    for record in records:

        metadata.append(
            {
                "chunk_id": record["chunk_id"],
                "document_id": record["document_id"],
                "text": record["text"],
                "metadata": record["metadata"],
            }
        )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():

    print(
        "\n========== BUILDING FAISS INDEX ==========\n"
    )

    records = load_embeddings()

    print(
        f"Embedding records: {len(records)}"
    )

    index = build_index(records)

    print(
        f"Vector dimension: {index.d}"
    )

    print(
        f"Vectors in index: {index.ntotal}"
    )

    save_index(index)

    save_metadata(records)

    print(
        "\n========== FAISS INDEX COMPLETE ==========\n"
    )

    print(
        f"Index: {INDEX_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


if __name__ == "__main__":
    main()