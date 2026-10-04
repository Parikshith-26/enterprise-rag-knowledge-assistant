from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


def main():

    evaluator = RetrievalEvaluator(top_k=5)

    results, metrics = evaluator.evaluate()

    print("\n========== RETRIEVAL EVALUATION ==========\n")

    for index, result in enumerate(results, start=1):

        print(f"{index}. {result['question']}")

        print(f"   Hit@1: {result['hit_at_1']}")
        print(f"   Hit@5: {result['hit_at_5']}")
        print(
            f"   Reciprocal Rank: "
            f"{result['reciprocal_rank']:.2f}"
        )

        # Show retrieved results for failed questions
        if not result["hit_at_5"]:

            print("\n   ---------- TOP RETRIEVED RESULTS ----------")

            retrieved = result.get(
                "retrieved_results",
                []
            )

            for rank, retrieved_result in enumerate(
                retrieved,
                start=1
            ):

                metadata = retrieved_result.get(
                    "metadata",
                    {}
                )

                print(f"\n   Rank {rank}")
                print(
                    f"   Document: "
                    f"{retrieved_result.get('document_id')}"
                )
                print(
                    f"   Section: "
                    f"{metadata.get('section')}"
                )
                print(
                    f"   Chunk: "
                    f"{retrieved_result.get('chunk_id')}"
                )
                print(
                    f"   Text: "
                    f"{retrieved_result.get('text', '')[:250]}"
                )

        print()

    print("========== METRICS ==========\n")

    print(
        f"Questions: {metrics['total_questions']}"
    )

    print(
        f"Hit@1: {metrics['hit_at_1']:.2%}"
    )

    print(
        f"Hit@5: {metrics['hit_at_5']:.2%}"
    )

    print(
        f"MRR: {metrics['mrr']:.3f}"
    )


if __name__ == "__main__":
    main()