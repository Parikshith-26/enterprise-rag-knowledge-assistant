from src.ingestion.pdf_loader import load_pdf


pdf_path = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank/pdf/Agriculture Loan.pdf"
)

document = load_pdf(
    pdf_path,
    company="ZX Bank"
)

print("Document ID:", document.document_id)
print("Title:", document.title)
print("Type:", document.document_type)
print("Company:", document.metadata["company"])
print("Pages:", document.metadata["page_count"])

print("\n--- EXTRACTED TEXT ---\n")
print(document.text[:3000])