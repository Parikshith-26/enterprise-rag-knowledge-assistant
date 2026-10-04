import json

from src.ingestion.base import Document
from src.chunking.chunker import chunk_document


with open(
    "data/processed/zx_bank_documents.jsonl",
    "r",
    encoding="utf-8"
) as file:

    data = json.loads(file.readline())


document = Document(
    document_id=data["document_id"],
    source=data["source"],
    title=data["title"],
    document_type=data["document_type"],
    text=data["text"],
    metadata=data["metadata"],
)


chunks = chunk_document(document)


print("\n========== CHUNKING TEST ==========\n")

print("Document:", document.title)
print("Original words:", len(document.text.split()))
print("Chunks created:", len(chunks))

for chunk in chunks:

    print("\n-----------------------------------")

    print("Chunk ID:", chunk.chunk_id)
    print("Word count:", chunk.metadata["word_count"])

    print("\nText:")
    print(chunk.text[:1000])