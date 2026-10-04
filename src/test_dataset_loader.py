from src.ingestion.dataset_loader import load_dataset


dataset_path = (
    "../RAG-Multi-Corpus-main/"
    "datasets/ZX Bank"
)

documents = load_dataset(
    dataset_path,
    company="ZX Bank"
)

print("\n========== DATASET INGESTION ==========\n")

print("Total documents:", len(documents))

print("\nDocument types:")

type_counts = {}

for document in documents:

    document_type = document.document_type

    type_counts[document_type] = (
        type_counts.get(document_type, 0) + 1
    )

for document_type, count in sorted(type_counts.items()):

    print(
        f"{document_type:10} : {count}"
    )

print("\nFirst 10 documents:")

for document in documents[:10]:

    print(
        f"{document.document_type:10} | "
        f"{document.title}"
    )