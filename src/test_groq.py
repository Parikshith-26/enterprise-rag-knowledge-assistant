from src.generation.groq_client import GroqClient


def main():
    llm = GroqClient()

    answer = llm.generate(
        system_prompt="You are a helpful AI assistant.",
        user_prompt="Explain Retrieval-Augmented Generation in 2 simple sentences.",
    )

    print("\n========== GROQ TEST ==========\n")
    print(answer)
    print("\n================================\n")


if __name__ == "__main__":
    main()