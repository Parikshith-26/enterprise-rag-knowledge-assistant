from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:

    def __init__(
        self,
        model_name: str = MODEL_NAME
    ):
        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        print("Embedding model loaded.")

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        if not text.strip():
            raise ValueError(
                "Cannot embed empty text."
            )

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return vectors.tolist()