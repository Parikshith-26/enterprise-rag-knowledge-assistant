import json
import re
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever


EVALUATION_FILE = Path(
    "data/evaluation/zx_bank_questions.json"
)


class RetrievalEvaluator:

    def __init__(self, top_k=5):
        self.retriever = HybridRetriever(top_k=top_k)
        self.top_k = top_k

    def load_questions(self):

        with EVALUATION_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # NORMALIZATION HELPERS
    # ---------------------------------------------------------

    def normalize_text(self, text):

        if not text:
            return ""

        text = str(text).lower().strip()

        # Normalize common punctuation
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = text.replace("_", " ")

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def normalize_document_name(self, document):

        document = self.normalize_text(document)

        # ZX Bank prefixes are representation-specific.
        # Treat:
        #   Agriculture Loan
        #   ZX Bank Agriculture Loan
        # as the same document.
        document = re.sub(
            r"^zx bank\s+",
            "",
            document
        )

        return document.strip()

    def document_matches(
        self,
        retrieved_document,
        expected_document
    ):

        retrieved = self.normalize_document_name(
            retrieved_document
        )

        expected = self.normalize_document_name(
            expected_document
        )

        return retrieved == expected

    def normalize_section(self, section):

        section = self.normalize_text(section)

        # Remove Markdown heading markers.
        # Examples:
        # ## Documents Required
        # ### Documents Required
        section = re.sub(
            r"^#+\s*",
            "",
            section
        )

        # Remove Markdown emphasis markers.
        # Examples:
        # *Documents Required
        # **Documents Required**
        section = re.sub(
            r"^\*+\s*",
            "",
            section
        )

        section = re.sub(
            r"\s*\*+$",
            "",
            section
        )

        # Remove decorative symbols/emojis at the beginning.
        # Example:
        # 📱 How to Activate UPI with ZX Bank
        section = re.sub(
            r"^[^\w]+",
            "",
            section
        )

        # Normalize common recommendation suffix.
        # Example:
        # Online Application (Recommended)
        section = re.sub(
            r"\s*\(recommended\)\s*$",
            "",
            section
        )

        # Normalize whitespace again.
        section = re.sub(
            r"\s+",
            " ",
            section
        )

        return section.strip()

    def section_matches(
        self,
        retrieved_section,
        expected_section
    ):

        if not expected_section:
            return True

        retrieved = self.normalize_section(
            retrieved_section
        )

        expected = self.normalize_section(
            expected_section
        )

        return retrieved == expected

    # ---------------------------------------------------------
    # QUESTION EVALUATION
    # ---------------------------------------------------------

    def evaluate_question(self, item):

        question = item["question"]

        expected_document = item[
            "expected_document"
        ]

        expected_section = item.get(
            "expected_section"
        )

        # Retrieve top-k results
        results = self.retriever.search(
            question
        )

        hit_at_1 = False
        hit_at_5 = False
        reciprocal_rank = 0.0

        # Check whether expected document/section
        # appears in retrieved results
        for rank, result in enumerate(
            results,
            start=1
        ):

            document_id = result.get(
                "document_id",
                ""
            )

            metadata = result.get(
                "metadata",
                {}
            )

            section = metadata.get(
                "section",
                ""
            )

            document_match = (
                self.document_matches(
                    document_id,
                    expected_document
                )
            )

            section_match = (
                self.section_matches(
                    section,
                    expected_section
                )
            )

            if document_match and section_match:

                if rank == 1:
                    hit_at_1 = True

                hit_at_5 = True

                reciprocal_rank = 1 / rank

                break

        return {
            "question": question,

            "expected_document":
                expected_document,

            "expected_section":
                expected_section,

            "hit_at_1":
                hit_at_1,

            "hit_at_5":
                hit_at_5,

            "reciprocal_rank":
                reciprocal_rank,

            # Keep retrieved results
            # for debugging failed questions.
            "retrieved_results":
                results,
        }

    # ---------------------------------------------------------
    # FULL EVALUATION
    # ---------------------------------------------------------

    def evaluate(self):

        questions = self.load_questions()

        results = []

        for item in questions:

            result = self.evaluate_question(
                item
            )

            results.append(result)

        total = len(results)

        if total == 0:

            return results, {
                "total_questions": 0,
                "hit_at_1": 0.0,
                "hit_at_5": 0.0,
                "mrr": 0.0,
            }

        hit_at_1 = sum(
            result["hit_at_1"]
            for result in results
        )

        hit_at_5 = sum(
            result["hit_at_5"]
            for result in results
        )

        mrr = sum(
            result["reciprocal_rank"]
            for result in results
        ) / total

        metrics = {

            "total_questions":
                total,

            "hit_at_1":
                hit_at_1 / total,

            "hit_at_5":
                hit_at_5 / total,

            "mrr":
                mrr,
        }

        return results, metrics


if __name__ == "__main__":
    pass