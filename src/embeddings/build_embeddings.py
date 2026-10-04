import json
from pathlib import Path

from src.chunking.chunker import chunk_document
from src.embeddings.embedder import Embedder
from src.ingestion.base import Document


INPUT_PATH = Path(
    "data/processed/zx_bank_documents.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/zx_bank_embeddings.json"
)


def load_documents() -> list[Document]:

    documents = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            data = json.loads(line)

            document = Document(
                document_id=data["document_id"],
                source=data["source"],
                title=data["title"],
                document_type=data["document_type"],
                text=data["text"],
                metadata=data["metadata"],
            )

            documents.append(document)

    return documents


def main():

    print("\n========== BUILDING EMBEDDINGS ==========\n")

    documents = load_documents()

    print(
        f"Documents loaded: {len(documents)}"
    )

    all_chunks = []

    for document in documents:

        chunks = chunk_document(document)

        all_chunks.extend(chunks)

    print(
        f"Chunks created: {len(all_chunks)}"
    )

    texts = [
        chunk.text
        for chunk in all_chunks
    ]

    print("\nLoading embedding model...")

    embedder = Embedder()

    print("\nGenerating embeddings...")

    vectors = embedder.model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    records = []

    for chunk, vector in zip(
        all_chunks,
        vectors
    ):

        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": vector.tolist(),
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False
        )

    print("\n========== EMBEDDINGS COMPLETE ==========\n")

    print(
        f"Records saved: {len(records)}"
    )

    print(
        f"Vector dimensions: {len(records[0]['embedding'])}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()