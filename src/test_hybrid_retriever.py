# src/test_hybrid_retriever.py

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)


def print_results(
    query: str,
    results: list[dict],
) -> None:

    print()
    print("=" * 70)
    print(
        f"QUERY: {query}"
    )
    print("=" * 70)

    if not results:

        print(
            "No results found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            "-" * 35
        )

        print(
            f"Rank: {rank}"
        )

        print(
            "Final ranking score: "
            f"{result.get('final_ranking_score', 0.0):.4f}"
        )

        print(
            "Reranker score: "
            f"{result.get('reranker_score', 0.0):.4f}"
        )

        print(
            "Hybrid score: "
            f"{result.get('hybrid_score', 0.0):.4f}"
        )

        print(
            "RRF score: "
            f"{result.get('rrf_score', 0.0):.4f}"
        )

        print(
            "Semantic score: "
            f"{result.get('semantic_score', 0.0):.4f}"
        )

        print(
            "Keyword score: "
            f"{result.get('bm25_score', 0.0):.4f}"
        )

        print(
            "Entity score: "
            f"{result.get('entity_score', 0.0):.4f}"
        )

        print(
            "Metadata score: "
            f"{result.get('metadata_score', 0.0):.4f}"
        )

        print(
            "Intent score: "
            f"{result.get('intent_score', 0.0):.4f}"
        )

        print(
            "Section match score: "
            f"{result.get('section_match_score', 0.0):.4f}"
        )

        print(
            f"Document: "
            f"{result.get('document_id', 'N/A')}"
        )

        print(
            f"Section: "
            f"{result.get('section', 'N/A')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id', 'N/A')}"
        )

        print()
        print("Text:")

        text = result.get(
            "text",
            "",
        )

        print(text)

    print()
    print("=" * 70)


def main():

    print(
        "\n========== HYBRID RETRIEVAL TEST =========="
    )

    retriever = HybridRetriever(
        top_k=5
    )

    queries = [
        "What documents are required for agriculture loan?",
        "How can I activate UPI with ZX Bank?",
        "Where can I find ZX Bank ATMs?",
    ]

    for query in queries:

        results = retriever.search(
            query
        )

        print_results(
            query,
            results,
        )

    print(
        "\n========== HYBRID RETRIEVAL TEST COMPLETE =========="
    )


if __name__ == "__main__":
    main()