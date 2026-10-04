import json
from pathlib import Path

import numpy as np


METADATA_PATH = Path(
    "data/indexes/zx_bank_metadata.json"
)


def main():

    print(
        "\n========== AGRICULTURE RETRIEVAL DIAGNOSTIC ==========\n"
    )

    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        top_k=20,
        candidate_k=20
    )

    query = (
        "What documents are required for agriculture loan?"
    )

    print(f"Query: {query}\n")

    # --------------------------------------------------
    # Get raw semantic scores
    # --------------------------------------------------

    semantic_scores = (
        retriever.semantic_search(query)
    )

    # --------------------------------------------------
    # Get raw BM25 scores
    # --------------------------------------------------

    keyword_scores = (
        retriever.keyword_search(query)
    )

    # --------------------------------------------------
    # Find every Agriculture Loan Documents Required
    # chunk
    # --------------------------------------------------

    target_indices = []

    for index, record in enumerate(
        retriever.metadata
    ):

        document_id = record.get(
            "document_id",
            ""
        )

        chunk_id = record.get(
            "chunk_id",
            ""
        )

        section = record.get(
            "metadata",
            {}
        ).get(
            "section",
            ""
        )

        if (
            "Agriculture Loan"
            in document_id
            and "chunk_4"
            in chunk_id
            and "Documents Required"
            in section
        ):
            target_indices.append(index)

    print(
        "Target Agriculture Loan "
        "'Documents Required' chunks:"
    )

    print(
        target_indices
    )

    # --------------------------------------------------
    # Print scores for exact target chunks
    # --------------------------------------------------

    for index in target_indices:

        record = retriever.metadata[index]

        print(
            "\n" + "=" * 70
        )

        print(
            f"FAISS index: {index}"
        )

        print(
            f"Document ID: "
            f"{record.get('document_id')}"
        )

        print(
            f"Chunk ID: "
            f"{record.get('chunk_id')}"
        )

        print(
            f"Section: "
            f"{record.get('metadata', {}).get('section')}"
        )

        print(
            f"Semantic raw score: "
            f"{semantic_scores.get(index, 'NOT IN TOP 20')}"
        )

        print(
            f"BM25 raw score: "
            f"{keyword_scores.get(index, 'NOT IN TOP 20')}"
        )

        print("\nText:")

        print(
            record.get("text", "")
        )

    # --------------------------------------------------
    # Show the top semantic candidates
    # --------------------------------------------------

    print(
        "\n\n========== TOP SEMANTIC CANDIDATES ==========\n"
    )

    semantic_sorted = sorted(
        semantic_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (index, score) in enumerate(
        semantic_sorted,
        start=1
    ):

        record = retriever.metadata[index]

        print(
            f"{rank}. "
            f"{score:.4f} | "
            f"{record.get('document_id')} | "
            f"{record.get('chunk_id')} | "
            f"{record.get('metadata', {}).get('section', 'N/A')}"
        )

    # --------------------------------------------------
    # Show the top BM25 candidates
    # --------------------------------------------------

    print(
        "\n\n========== TOP BM25 CANDIDATES ==========\n"
    )

    keyword_sorted = sorted(
        keyword_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (index, score) in enumerate(
        keyword_sorted,
        start=1
    ):

        record = retriever.metadata[index]

        print(
            f"{rank}. "
            f"{score:.4f} | "
            f"{record.get('document_id')} | "
            f"{record.get('chunk_id')} | "
            f"{record.get('metadata', {}).get('section', 'N/A')}"
        )

    print(
        "\n========== DIAGNOSTIC COMPLETE ==========\n"
    )


if __name__ == "__main__":
    main()