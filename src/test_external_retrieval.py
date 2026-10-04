from pathlib import Path

import faiss
import numpy as np

from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_document
from src.embeddings.embedder import Embedder


PDF_PATH = Path(
    "data/documents/test_external/enterprise_rag_test_document.pdf"
)

COMPANY = "Nimbus Analytics"


def main():
    print("\n========== EXTERNAL PDF RETRIEVAL TEST ==========\n")

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    document = load_document(PDF_PATH, COMPANY)

    print(f"Document: {document.title}")
    print(f"Characters: {len(document.text)}")

    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    chunks = chunk_document(
        document,
        max_words=350,
        overlap_words=50
    )

    print(f"Chunks: {len(chunks)}")

    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    embedder = Embedder()

    texts = [chunk.text for chunk in chunks]

    print("\nCreating embeddings...")

    embeddings = embedder.model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------
    # 4. Build temporary FAISS index
    # --------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS vectors: {index.ntotal}")

    # --------------------------------------------------
    # 5. Questions
    # --------------------------------------------------

    questions = [
        "What authentication is required to access Nimbus Analytics?",
        "What can an Analyst do?",
        "How often is customer activity data refreshed?",
        "Who can manage permissions?",
        "Where should urgent access incidents be reported?",
    ]

    # --------------------------------------------------
    # 6. Retrieve relevant chunks
    # --------------------------------------------------

    print("\n========== RETRIEVAL RESULTS ==========\n")

    for question in questions:

        print("=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        query_embedding = embedder.model.encode(
            [question],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = index.search(
            query_embedding,
            k=min(2, len(chunks))
        )

        for rank, (score, chunk_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            chunk = chunks[chunk_index]

            print("\n" + "-" * 70)
            print(f"Rank: {rank}")
            print(f"Score: {score:.4f}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Words: {len(chunk.text.split())}")
            print("\nRetrieved text:")
            print(chunk.text)

    print("\n")
    print("=" * 80)
    print("EXTERNAL RETRIEVAL TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()