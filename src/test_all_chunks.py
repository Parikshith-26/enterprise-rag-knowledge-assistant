import json

from src.ingestion.base import Document
from src.chunking.chunker import chunk_document


INPUT_PATH = "data/processed/zx_bank_documents.jsonl"


def main():

    documents = []

    with open(
        INPUT_PATH,
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

    all_chunks = []

    for document in documents:

        chunks = chunk_document(document)

        all_chunks.extend(chunks)

    print("\n========== CORPUS CHUNKING ==========\n")

    print(
        "Documents:",
        len(documents)
    )

    print(
        "Total chunks:",
        len(all_chunks)
    )

    print(
        "Average chunks/document:",
        round(
            len(all_chunks) / len(documents),
            2
        )
    )

    print("\nChunks by document type:")

    counts = {}

    for chunk in all_chunks:

        document_type = chunk.metadata[
            "document_type"
        ]

        counts[document_type] = (
            counts.get(document_type, 0) + 1
        )

    for document_type, count in sorted(
        counts.items()
    ):

        print(
            f"{document_type:10} : {count}"
        )

    print("\nFirst 10 chunks:")

    for chunk in all_chunks[:10]:

        print("\n-----------------------------------")

        print(
            "Chunk:",
            chunk.chunk_id
        )

        print(
            "Document:",
            chunk.metadata["title"]
        )

        print(
            "Section:",
            chunk.metadata.get(
                "section",
                "None"
            )
        )

        print(
            "Words:",
            chunk.metadata["word_count"]
        )


if __name__ == "__main__":
    main()