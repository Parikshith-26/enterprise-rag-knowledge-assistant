import json
from pathlib import Path

from src.ingestion.dataset_loader import load_dataset


DATASET_PATH = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank"
)

OUTPUT_PATH = Path("data/processed/zx_bank_documents.jsonl")


def document_to_dict(document):
    return {
        "document_id": document.document_id,
        "source": document.source,
        "title": document.title,
        "document_type": document.document_type,
        "text": document.text,
        "metadata": document.metadata,
    }


def main():
    print("Starting ZX Bank ingestion...")

    documents = load_dataset(
        DATASET_PATH,
        company="ZX Bank"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        for document in documents:
            json.dump(
                document_to_dict(document),
                file,
                ensure_ascii=False
            )

            file.write("\n")

    print()
    print("========== INGESTION COMPLETE ==========")
    print(f"Documents processed: {len(documents)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()