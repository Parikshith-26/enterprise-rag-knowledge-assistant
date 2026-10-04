from src.retrieval.retriever import Retriever


def main():

    print("\n========== RETRIEVAL TEST ==========\n")

    retriever = Retriever(
        top_k=5
    )

    query = (
        "What documents are required for agriculture loan?"
    )

    print(
        f"Query: {query}\n"
    )

    results = retriever.search(
        query
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            "\n-----------------------------------"
        )

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Document: "
            f"{result['metadata']['title']}"
        )

        print(
            f"Section: "
            f"{result['metadata'].get('section', 'N/A')}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print("\nText:")

        print(
            result["text"][:1000]
        )


if __name__ == "__main__":
    main()