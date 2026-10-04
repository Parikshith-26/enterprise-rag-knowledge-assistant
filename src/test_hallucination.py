from src.generation.answer_generator import AnswerGenerator


def main():

    rag = AnswerGenerator(top_k=5)

    question = (
        "What is the exact interest rate for "
        "a ZX Bank Agriculture Loan?"
    )

    result = rag.generate_answer(question)

    print("\n========== HALLUCINATION TEST ==========\n")

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\n========== RETRIEVED SOURCES ==========\n")

    for index, source in enumerate(
        result["sources"],
        start=1
    ):

        print(f"[{index}] {source['document_id']}")
        print(f"    Section: {source['section']}")
        print(f"    Chunk: {source['chunk_id']}")
        print()


if __name__ == "__main__":
    main()