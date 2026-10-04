import os

from dotenv import load_dotenv
from groq import Groq


# Load variables from .env
load_dotenv()


class GroqClient:
    """
    Wrapper around the Groq API.

    Handles communication with the LLM.
    RAG retrieval logic remains separate.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 1024,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )

        return response.choices[0].message.content