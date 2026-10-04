from src.generation.answer_generator import AnswerGenerator


def main():

    rag = AnswerGenerator(top_k=5)

    question = "What documents are required for an Agriculture Loan?"

    result = rag.generate_answer(question)

    print("\n========== RAG ANSWER ==========\n")
    print(result["answer"])

    print("\n========== SOURCES ==========\n")

    for index, source in enumerate(result["sources"], start=1):

        print(f"[{index}] {source['document_id']}")
        print(f"    Section: {source['section']}")
        print(f"    Chunk: {source['chunk_id']}")
        print(f"    Type: {source['document_type']}")
        print()


if __name__ == "__main__":
    main()