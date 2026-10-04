from src.ingestion.markdown_loader import load_markdown


markdown_path = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank/md/Agriculture Loan.md"
)

document = load_markdown(
    markdown_path,
    company="ZX Bank"
)

print("Document ID:", document.document_id)
print("Title:", document.title)
print("Type:", document.document_type)
print("Company:", document.metadata["company"])

print("\n--- EXTRACTED TEXT ---\n")
print(document.text[:3000])