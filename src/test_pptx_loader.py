from src.ingestion.pptx_loader import load_pptx


pptx_path = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank/pptx/Agriculture Loan.pptx"
)

document = load_pptx(
    pptx_path,
    company="ZX Bank"
)

print("Document ID:", document.document_id)
print("Title:", document.title)
print("Type:", document.document_type)
print("Company:", document.metadata["company"])
print("Slides:", document.metadata["slide_count"])

print("\n--- EXTRACTED TEXT ---\n")
print(document.text[:3000])