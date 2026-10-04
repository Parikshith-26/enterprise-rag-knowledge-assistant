from src.embeddings.embedder import Embedder


def main():

    print("\n========== EMBEDDING TEST ==========\n")

    embedder = Embedder()

    text = """
    Unified Payments Interface (UPI) is a real-time
    payment system that enables instant money transfer
    between bank accounts using a mobile device.
    """

    vector = embedder.embed_text(text)

    print()
    print("Vector type:", type(vector))
    print("Vector dimensions:", len(vector))

    print("\nFirst 10 values:")

    for value in vector[:10]:
        print(value)


if __name__ == "__main__":
    main()