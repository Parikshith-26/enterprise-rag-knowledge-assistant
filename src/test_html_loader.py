from src.ingestion.html_loader import load_html


html_path = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank/html/Agriculture Loan.html"
)

document = load_html(
    html_path,
    company="ZX Bank"
)

print("Document ID:", document.document_id)
print("Title:", document.title)
print("Type:", document.document_type)
print("Company:", document.metadata["company"])

print("\n--- EXTRACTED TEXT ---\n")
print(document.text[:3000])