import json
import re
import unicodedata
from pathlib import Path

from src.generation.answer_generator import AnswerGenerator


EVALUATION_FILE = Path(
    "data/evaluation/zx_bank_answer_questions.json"
)


def normalize_text(text: str) -> str:
    """
    Normalize text before keyword comparison.

    Handles:
    - capitalization
    - Unicode characters
    - different hyphen types
    - punctuation
    - repeated whitespace
    """

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    # Normalize different dash/hyphen characters
    text = re.sub(
        r"[-‐-‒–—―]",
        " ",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


class AnswerEvaluator:

    def __init__(self, top_k=5):
        self.rag = AnswerGenerator(
            top_k=top_k
        )

    def load_questions(self):

        with EVALUATION_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    def evaluate_question(self, item):

        question = item["question"]

        expected_keywords = item[
            "expected_keywords"
        ]

        result = self.rag.generate_answer(
            question
        )

        answer = result["answer"]

        normalized_answer = normalize_text(
            answer
        )

        matched_keywords = []
        missing_keywords = []

        for keyword in expected_keywords:

            normalized_keyword = normalize_text(
                keyword
            )

            if normalized_keyword in normalized_answer:

                matched_keywords.append(
                    keyword
                )

            else:

                missing_keywords.append(
                    keyword
                )

        total_keywords = len(
            expected_keywords
        )

        coverage = (
            len(matched_keywords)
            / total_keywords
            if total_keywords > 0
            else 0.0
        )

        return {
            "question": question,
            "answer": answer,
            "expected_keywords": expected_keywords,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "coverage": coverage,
            "sources": result["sources"],
        }

    def evaluate(self):

        questions = self.load_questions()

        results = []

        for item in questions:

            print(
                f"\nEvaluating: "
                f"{item['question']}"
            )

            result = self.evaluate_question(
                item
            )

            results.append(result)

        total = len(results)

        average_coverage = (
            sum(
                result["coverage"]
                for result in results
            )
            / total
            if total > 0
            else 0.0
        )

        return results, {
            "total_questions": total,
            "average_keyword_coverage":
                average_coverage,
        }