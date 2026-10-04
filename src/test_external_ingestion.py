from pathlib import Path

from src.ingestion.loader import load_document


PDF_PATH = Path(
    "data/documents/test_external/enterprise_rag_test_document.pdf"
)

COMPANY = "Nimbus Analytics"


def main():
    print("\n========== EXTERNAL PDF TEST ==========\n")

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    print(f"Loading: {PDF_PATH}")

    document = load_document(
        PDF_PATH,
        COMPANY
    )

    print("\n========== DOCUMENT LOADED ==========\n")

    print(f"Document ID: {document.document_id}")
    print(f"Title: {document.title}")
    print(f"Type: {document.document_type}")
    print(f"Source: {document.source}")
    print(f"Characters: {len(document.text)}")

    print("\n---------- PREVIEW ----------\n")
    print(document.text[:3000])

    print("\n========== TEST COMPLETE ==========\n")


if __name__ == "__main__":
    main()