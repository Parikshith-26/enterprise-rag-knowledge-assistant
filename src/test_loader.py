from src.ingestion.loader import load_document


files = [
    (
        "../RAG-Multi-Corpus-main/"
        "datasets/ZX Bank/pdf/Agriculture Loan.pdf"
    ),
    (
        "../RAG-Multi-Corpus-main/"
        "datasets/ZX Bank/docx/Agriculture Loan.docx"
    ),
    (
        "../RAG-Multi-Corpus-main/"
        "datasets/ZX Bank/html/Agriculture Loan.html"
    ),
    (
        "../RAG-Multi-Corpus-main/"
        "datasets/ZX Bank/md/Agriculture Loan.md"
    ),
    (
        "../RAG-Multi-Corpus-main/"
        "datasets/ZX Bank/pptx/Agriculture Loan.pptx"
    ),
]


for file_path in files:

    document = load_document(
        file_path,
        company="ZX Bank"
    )

    print(
        f"{document.document_type.upper():10} | "
        f"{document.title:20} | "
        f"{len(document.text):6} characters"
    )