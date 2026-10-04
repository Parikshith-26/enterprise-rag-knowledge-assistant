from pathlib import Path

from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_document


PDF_PATH = Path(
    "data/documents/test_external/enterprise_rag_test_document.pdf"
)

COMPANY = "Nimbus Analytics"


def main():

    print("\n========== EXTERNAL PDF CHUNK TEST ==========\n")

    document = load_document(
        PDF_PATH,
        COMPANY
    )

    print(f"Document: {document.title}")
    print(f"Characters: {len(document.text)}")

    chunks = chunk_document(
        document,
        max_words=350,
        overlap_words=50
    )

    print(f"\nTotal chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 70)

        print(f"Chunk {index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Words: {len(chunk.text.split())}")

        print("\nText:")
        print(chunk.text[:1000])

    print("\n" + "=" * 70)
    print("EXTERNAL CHUNK TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()