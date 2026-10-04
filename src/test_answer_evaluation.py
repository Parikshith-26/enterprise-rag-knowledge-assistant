from src.evaluation.answer_evaluator import (
    AnswerEvaluator
)


def main():

    evaluator = AnswerEvaluator(top_k=5)

    results, metrics = evaluator.evaluate()

    print(
        "\n========== ANSWER EVALUATION ==========\n"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. {result['question']}"
        )

        print(
            f"   Coverage: "
            f"{result['coverage']:.2%}"
        )

        print(
            f"   Matched: "
            f"{result['matched_keywords']}"
        )

        print(
            f"   Missing: "
            f"{result['missing_keywords']}"
        )

        print(
            f"   Answer:\n"
            f"   {result['answer']}"
        )

        print()

    print(
        "========== FINAL METRICS ==========\n"
    )

    print(
        f"Questions: "
        f"{metrics['total_questions']}"
    )

    print(
        f"Average Keyword Coverage: "
        f"{metrics['average_keyword_coverage']:.2%}"
    )


if __name__ == "__main__":
    main()