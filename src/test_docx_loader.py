from src.ingestion.docx_loader import load_docx


docx_path = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank/docx/Agriculture Loan.docx"
)

document = load_docx(
    docx_path,
    company="ZX Bank"
)

print("Document ID:", document.document_id)
print("Title:", document.title)
print("Type:", document.document_type)
print("Company:", document.metadata["company"])

print("\n--- EXTRACTED TEXT ---\n")
print(document.text[:3000])