from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class HybridRetriever:
    """
    Hybrid enterprise retriever.

    Pipeline:

        Query normalization
                ↓
        Intent detection
                ↓
        FAISS semantic retrieval
                +
        BM25 keyword retrieval
                +
        Intent-aware lexical retrieval
                ↓
        Candidate merge
                ↓
        Reciprocal Rank Fusion
                ↓
        Metadata / entity / phrase scoring
                ↓
        Intent scoring
                ↓
        Explicit section matching
                ↓
        Cross-encoder reranking
                ↓
        Final ranking
                ↓
        Document/section diversity
                ↓
        Final Top-K
    """

    # =========================================================
    # CONFIGURATION
    # =========================================================

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    RERANKER_THRESHOLD = 2.0

    DEFAULT_TOP_K = 5
    DEFAULT_CANDIDATE_K = 50
    DEFAULT_RRF_K = 60

    MAX_RESULTS_PER_DOCUMENT = 2
    NEAR_DUPLICATE_THRESHOLD = 0.90

    # Normal hybrid boosts
    SECTION_MATCH_BOOST = 0.30
    ENTITY_MATCH_BOOST = 0.05
    PHRASE_MATCH_BOOST = 0.25

    # Final ranking
    RERANKER_WEIGHT = 0.55
    INTENT_WEIGHT = 0.20
    SECTION_WEIGHT = 0.20
    METADATA_WEIGHT = 0.05

    # Intent-specific candidate retrieval
    INTENT_CANDIDATE_LIMIT = 100

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        project_root = Path(__file__).resolve().parents[2]

        self.index_path = (
            project_root
            / "data"
            / "indexes"
            / "zx_bank.index"
        )

        self.metadata_path = (
            project_root
            / "data"
            / "indexes"
            / "zx_bank_metadata.json"
        )

        self.embedding_path = (
            project_root
            / "data"
            / "processed"
            / "zx_bank_embeddings.json"
        )

        # -----------------------------------------------------
        # FAISS
        # -----------------------------------------------------

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found: {self.metadata_path}"
            )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        # -----------------------------------------------------
        # Embedding records
        # -----------------------------------------------------

        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f"Embedding records not found: {self.embedding_path}"
            )

        with open(
            self.embedding_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.embedding_records = json.load(file)

        # -----------------------------------------------------
        # Embedding model
        # -----------------------------------------------------

        self.embedding_model = SentenceTransformer(
            self.EMBEDDING_MODEL
        )

        # -----------------------------------------------------
        # Cross encoder
        # -----------------------------------------------------

        self.reranker = CrossEncoder(
            self.RERANKER_MODEL
        )

        # -----------------------------------------------------
        # BM25
        # -----------------------------------------------------

        self.bm25_documents = []

        for record in self.embedding_records:
            metadata = record.get("metadata", {})

            document_id = record.get(
                "document_id",
                metadata.get("document_id", ""),
            )

            section = record.get(
                "section",
                metadata.get("section", ""),
            )

            text = record.get("text", "")

            searchable_text = " ".join(
                [
                    str(document_id),
                    str(section),
                    str(text),
                ]
            )

            self.bm25_documents.append(
                self.tokenize(searchable_text)
            )

        self.bm25 = BM25Okapi(
            self.bm25_documents
        )

    # =========================================================
    # TEXT UTILITIES
    # =========================================================

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not text:
            return []

        text = str(text).lower()

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text,
        )

    @staticmethod
    def normalize_for_duplicate(text: str) -> str:
        if not text:
            return ""

        text = str(text).lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"[^a-z0-9 ]",
            "",
            text,
        )

        return text.strip()

    # =========================================================
    # QUERY NORMALIZATION
    # =========================================================

    def normalize_query(self, query: str) -> str:
        if not query:
            return ""

        query = query.lower().strip()

        replacements = {
            "atms": "atm",
            "loans": "loan",
            "accounts": "account",
            "documents": "document",
            "locations": "location",
            "applications": "application",
            "branches": "branch",
            "cheques": "cheque",
            "payments": "payment",
            "pumps": "pump",
            "stations": "station",
        }

        for source, target in replacements.items():
            query = re.sub(
                rf"\b{re.escape(source)}\b",
                target,
                query,
            )

        return query

    # =========================================================
    # INTENT DETECTION
    # =========================================================

    def detect_intent(self, query: str) -> str:
        query = self.normalize_query(query)

        # -----------------------------------------------------
        # VERY SPECIFIC INTENTS FIRST
        # -----------------------------------------------------

        # UPI activation MUST come before generic activation.
        if (
            "upi" in query
            and any(
                phrase in query
                for phrase in [
                    "activate",
                    "activation",
                    "enable",
                    "register",
                    "how to",
                    "how can i",
                    "how do i",
                ]
            )
        ):
            return "upi_activation"

        # Petrol pump ATM MUST come before generic location.
        if (
            "petrol pump" in query
            or (
                "petrol" in query
                and "pump" in query
            )
        ):
            return "petrol_pump_atm"

        # -----------------------------------------------------
        # DOCUMENTS REQUIRED
        # -----------------------------------------------------

        if any(
            phrase in query
            for phrase in [
                "document required",
                "document requirement",
                "required document",
                "documents needed",
                "document needed",
                "what document",
                "which document",
                "document need",
            ]
        ):
            return "documents_required"

        # -----------------------------------------------------
        # ELIGIBILITY
        # -----------------------------------------------------

        if any(
            phrase in query
            for phrase in [
                "who can apply",
                "eligible",
                "eligibility",
                "eligible for",
                "who is eligible",
            ]
        ):
            return "eligibility"

        # -----------------------------------------------------
        # APPLICATION
        # -----------------------------------------------------

        if any(
            phrase in query
            for phrase in [
                "how to apply",
                "apply online",
                "online application",
                "application",
                "how can i apply",
                "how do i apply",
            ]
        ):
            return "application"

        # -----------------------------------------------------
        # ACTIVATION
        # -----------------------------------------------------

        if any(
            phrase in query
            for phrase in [
                "activate",
                "activation",
                "enable",
                "register",
            ]
        ):
            return "activation"

        # -----------------------------------------------------
        # LOCATION
        # -----------------------------------------------------

        if any(
            phrase in query
            for phrase in [
                "location",
                "where can i find",
                "where is",
                "atm",
                "nearest",
            ]
        ):
            return "location"

        # -----------------------------------------------------
        # DEFINITION
        # -----------------------------------------------------

        if any(
            phrase in query
            for phrase in [
                "what is",
                "meaning",
                "define",
                "definition",
            ]
        ):
            return "definition"

        return "general"

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================

    def semantic_search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        normalized_query = self.normalize_query(query)

        if not normalized_query:
            return []

        query_embedding = self.embedding_model.encode(
            [normalized_query],
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_embedding,
            self.candidate_k,
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            if index_position >= len(
                self.embedding_records
            ):
                continue

            record = self.embedding_records[
                index_position
            ]

            result = self.build_result(
                record=record,
                semantic_score=float(score),
                semantic_rank=len(results) + 1,
            )

            results.append(result)

        return results

    # =========================================================
    # BM25 SEARCH
    # =========================================================

    def keyword_search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        normalized_query = self.normalize_query(query)

        query_tokens = self.tokenize(
            normalized_query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index_position in ranked_indices[
            : self.candidate_k
        ]:
            record = self.embedding_records[
                index_position
            ]

            result = self.build_result(
                record=record,
                bm25_score=float(
                    scores[index_position]
                ),
                bm25_rank=len(results) + 1,
            )

            results.append(result)

        return results

    # =========================================================
    # INTENT-AWARE LEXICAL SEARCH
    # =========================================================

    def intent_candidate_search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Search ALL chunks for intent-specific terminology.

        This is the important fix.

        It prevents cases such as:

            petrol pump ATM
                ↓
            railway/bus/park ATM

        and:

            activate UPI
                ↓
            activate mobile banking

        from being lost before reranking.
        """

        normalized_query = self.normalize_query(query)
        intent = self.detect_intent(
            normalized_query
        )

        query_tokens = set(
            self.tokenize(normalized_query)
        )

        if not query_tokens:
            return []

        scored = []

        for index_position, record in enumerate(
            self.embedding_records
        ):
            metadata = record.get(
                "metadata",
                {},
            )

            document_id = str(
                record.get(
                    "document_id",
                    metadata.get(
                        "document_id",
                        "",
                    ),
                )
            )

            section = str(
                record.get(
                    "section",
                    metadata.get(
                        "section",
                        "",
                    ),
                )
            )

            text = str(
                record.get(
                    "text",
                    "",
                )
            )

            document_lower = document_id.lower()
            section_lower = section.lower()
            text_lower = text.lower()

            score = 0.0

            # -------------------------------------------------
            # Generic token overlap
            # -------------------------------------------------

            combined_tokens = set(
                self.tokenize(
                    " ".join(
                        [
                            document_id,
                            section,
                            text,
                        ]
                    )
                )
            )

            overlap = (
                query_tokens
                & combined_tokens
            )

            if query_tokens:
                score += (
                    len(overlap)
                    / len(query_tokens)
                ) * 1.0

            # -------------------------------------------------
            # Intent-specific scoring
            # -------------------------------------------------

            if intent == "documents_required":
                if (
                    "documents required"
                    in section_lower
                ):
                    score += 10.0

                if (
                    "required documents"
                    in section_lower
                ):
                    score += 10.0

                if (
                    "documents required"
                    in text_lower
                ):
                    score += 3.0

                if (
                    "document required"
                    in text_lower
                ):
                    score += 2.0

            elif intent == "eligibility":
                if (
                    "who can apply"
                    in section_lower
                ):
                    score += 10.0

                if (
                    "eligibility"
                    in section_lower
                ):
                    score += 10.0

                if (
                    "who can apply"
                    in text_lower
                ):
                    score += 4.0

            elif intent == "application":
                if (
                    "online application"
                    in section_lower
                ):
                    score += 10.0

                if (
                    "how to apply"
                    in section_lower
                ):
                    score += 10.0

                if (
                    "application"
                    in section_lower
                ):
                    score += 5.0

                if (
                    "apply online"
                    in text_lower
                ):
                    score += 4.0

            elif intent == "upi_activation":
                if "upi" in document_lower:
                    score += 8.0

                if "upi" in section_lower:
                    score += 12.0

                if "upi" in text_lower:
                    score += 5.0

                if (
                    "activate"
                    in section_lower
                    or "activation"
                    in section_lower
                ):
                    score += 8.0

                if (
                    "activate upi"
                    in text_lower
                ):
                    score += 12.0

            elif intent == "petrol_pump_atm":
                if (
                    "petrol"
                    in document_lower
                ):
                    score += 8.0

                if (
                    "pump"
                    in document_lower
                ):
                    score += 8.0

                if (
                    "petrol"
                    in section_lower
                ):
                    score += 8.0

                if (
                    "pump"
                    in section_lower
                ):
                    score += 8.0

                if (
                    "petrol pump"
                    in text_lower
                ):
                    score += 12.0

                if "atm" in document_lower:
                    score += 4.0

                if "atm" in section_lower:
                    score += 4.0

            elif intent == "activation":
                if (
                    "activation"
                    in section_lower
                ):
                    score += 8.0

                if (
                    "activate"
                    in section_lower
                ):
                    score += 8.0

                if (
                    "registration"
                    in section_lower
                ):
                    score += 5.0

            elif intent == "location":
                if (
                    "location"
                    in section_lower
                ):
                    score += 7.0

                if "atm" in section_lower:
                    score += 6.0

                if "location" in text_lower:
                    score += 3.0

            if score > 0:
                result = self.build_result(
                    record=record
                )

                result[
                    "intent_candidate_score"
                ] = score

                result[
                    "intent_candidate_rank"
                ] = index_position

                scored.append(result)

        scored.sort(
            key=lambda item: item.get(
                "intent_candidate_score",
                0.0,
            ),
            reverse=True,
        )

        return scored[
            : self.INTENT_CANDIDATE_LIMIT
        ]

    # =========================================================
    # RESULT BUILDER
    # =========================================================

    def build_result(
        self,
        record: dict[str, Any],
        semantic_score: float = 0.0,
        semantic_rank: int | None = None,
        bm25_score: float = 0.0,
        bm25_rank: int | None = None,
    ) -> dict[str, Any]:

        metadata = record.get(
            "metadata",
            {},
        )

        return {
            "chunk_id": record.get(
                "chunk_id",
                record.get("id", ""),
            ),
            "document_id": record.get(
                "document_id",
                metadata.get(
                    "document_id",
                    "",
                ),
            ),
            "text": record.get(
                "text",
                "",
            ),
            "metadata": metadata,
            "source": record.get(
                "source",
                metadata.get(
                    "source",
                    "",
                ),
            ),
            "document_type": record.get(
                "document_type",
                metadata.get(
                    "document_type",
                    "",
                ),
            ),
            "section": metadata.get(
                "section",
                record.get(
                    "section",
                    "N/A",
                ),
            ),
            "semantic_score": semantic_score,
            "semantic_rank": semantic_rank,
            "bm25_score": bm25_score,
            "bm25_rank": bm25_rank,
        }

    # =========================================================
    # CHUNK KEY
    # =========================================================

    def get_chunk_key(
        self,
        result: dict[str, Any],
    ) -> str:

        chunk_id = str(
            result.get(
                "chunk_id",
                "",
            )
        )

        document_id = str(
            result.get(
                "document_id",
                "",
            )
        )

        if document_id or chunk_id:
            return (
                f"{document_id}::{chunk_id}"
            )

        return self.normalize_for_duplicate(
            result.get(
                "text",
                "",
            )
        )

    # =========================================================
    # MERGE CANDIDATES
    # =========================================================

    def merge_candidates(
        self,
        *result_lists: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        candidates = {}

        for result_list in result_lists:
            for result in result_list:
                key = self.get_chunk_key(
                    result
                )

                if key not in candidates:
                    candidates[key] = result
                    continue

                existing = candidates[key]

                existing["bm25_score"] = max(
                    existing.get(
                        "bm25_score",
                        0.0,
                    ),
                    result.get(
                        "bm25_score",
                        0.0,
                    ),
                )

                if result.get(
                    "bm25_rank"
                ) is not None:
                    existing["bm25_rank"] = (
                        result["bm25_rank"]
                    )

                existing["semantic_score"] = max(
                    existing.get(
                        "semantic_score",
                        0.0,
                    ),
                    result.get(
                        "semantic_score",
                        0.0,
                    ),
                )

                if result.get(
                    "semantic_rank"
                ) is not None:
                    if existing.get(
                        "semantic_rank"
                    ) is None:
                        existing[
                            "semantic_rank"
                        ] = result[
                            "semantic_rank"
                        ]

                existing[
                    "intent_candidate_score"
                ] = max(
                    existing.get(
                        "intent_candidate_score",
                        0.0,
                    ),
                    result.get(
                        "intent_candidate_score",
                        0.0,
                    ),
                )

        return list(
            candidates.values()
        )

    # =========================================================
    # RRF
    # =========================================================

    def reciprocal_rank_fusion(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        for result in candidates:
            score = 0.0

            semantic_rank = result.get(
                "semantic_rank"
            )

            bm25_rank = result.get(
                "bm25_rank"
            )

            if semantic_rank is not None:
                score += (
                    1.0
                    / (
                        self.rrf_k
                        + semantic_rank
                    )
                )

            if bm25_rank is not None:
                score += (
                    1.0
                    / (
                        self.rrf_k
                        + bm25_rank
                    )
                )

            result["rrf_score"] = score

        candidates.sort(
            key=lambda item: item.get(
                "rrf_score",
                0.0,
            ),
            reverse=True,
        )

        return candidates

    # =========================================================
    # ENTITY SCORE
    # =========================================================

    def entity_score(
        self,
        query: str,
        result: dict[str, Any],
    ) -> float:

        query_tokens = set(
            self.tokenize(
                self.normalize_query(query)
            )
        )

        if not query_tokens:
            return 0.0

        document_id = (
            str(
                result.get(
                    "document_id",
                    "",
                )
            )
        )

        section = (
            str(
                result.get(
                    "section",
                    "",
                )
            )
        )

        text = (
            str(
                result.get(
                    "text",
                    "",
                )
            )
        )

        combined = (
            f"{document_id} "
            f"{section} "
            f"{text}"
        )

        combined_tokens = set(
            self.tokenize(combined)
        )

        overlap = (
            query_tokens
            & combined_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )

    # =========================================================
    # PHRASE SCORE
    # =========================================================

    def phrase_match_score(
        self,
        query: str,
        result: dict[str, Any],
    ) -> float:

        normalized_query = (
            self.normalize_query(query)
        )

        if not normalized_query:
            return 0.0

        section = str(
            result.get(
                "section",
                "",
            )
        )

        document_id = str(
            result.get(
                "document_id",
                "",
            )
        )

        text = str(
            result.get(
                "text",
                "",
            )
        )

        combined = " ".join(
            [
                document_id,
                section,
                text,
            ]
        ).lower()

        combined = re.sub(
            r"\s+",
            " ",
            combined,
        ).strip()

        query_text = (
            normalized_query.lower()
        )

        # Strong exact phrase match.
        if query_text in combined:
            return 1.0

        stop_words = {
            "what",
            "where",
            "when",
            "who",
            "how",
            "can",
            "could",
            "would",
            "should",
            "i",
            "me",
            "my",
            "the",
            "a",
            "an",
            "is",
            "are",
            "for",
            "to",
            "of",
            "with",
            "and",
            "do",
            "does",
            "can",
        }

        tokens = self.tokenize(
            normalized_query
        )

        meaningful_tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

        if len(meaningful_tokens) < 2:
            return 0.0

        phrase_matches = 0
        total_phrases = 0

        for size in (3, 2):
            if len(meaningful_tokens) < size:
                continue

            for index in range(
                len(meaningful_tokens)
                - size
                + 1
            ):
                phrase = " ".join(
                    meaningful_tokens[
                        index:index + size
                    ]
                )

                total_phrases += 1

                if phrase in combined:
                    phrase_matches += 1

        if total_phrases == 0:
            return 0.0

        return (
            phrase_matches
            / total_phrases
        )

    # =========================================================
    # METADATA SCORE
    # =========================================================

    def metadata_score(
        self,
        query: str,
        result: dict[str, Any],
    ) -> float:

        query_tokens = set(
            self.tokenize(
                self.normalize_query(query)
            )
        )

        if not query_tokens:
            return 0.0

        metadata = result.get(
            "metadata",
            {},
        )

        document_id = str(
            result.get(
                "document_id",
                "",
            )
        )

        section = str(
            result.get(
                "section",
                "",
            )
        )

        title = str(
            metadata.get(
                "title",
                "",
            )
        )

        combined = (
            f"{document_id} "
            f"{section} "
            f"{title}"
        )

        metadata_tokens = set(
            self.tokenize(combined)
        )

        overlap = (
            query_tokens
            & metadata_tokens
        )

        return (
            len(overlap)
            / len(query_tokens)
        )

    # =========================================================
    # INTENT SCORE
    # =========================================================

    def intent_score(
        self,
        query: str,
        result: dict[str, Any],
    ) -> float:

        intent = self.detect_intent(
            query
        )

        section = self.normalize_for_duplicate(
            str(
                result.get(
                    "section",
                    "",
                )
            )
        )

        document = self.normalize_for_duplicate(
            str(
                result.get(
                    "document_id",
                    "",
                )
            )
        )

        text = self.normalize_for_duplicate(
            str(
                result.get(
                    "text",
                    "",
                )
            )
        )

        combined = (
            f"{section} "
            f"{document} "
            f"{text}"
        )

        # -----------------------------------------------------
        # Documents required
        # -----------------------------------------------------

        if intent == "documents_required":
            if section == "documents required":
                return 1.0

            if "documents required" in section:
                return 1.0

            if "required documents" in section:
                return 1.0

            # Text alone is intentionally weaker.
            if "documents required" in text:
                return 0.35

            if "required documents" in text:
                return 0.35

            if "required document" in text:
                return 0.25

            return 0.0

        # -----------------------------------------------------
        # Eligibility
        # -----------------------------------------------------

        if intent == "eligibility":
            if section in {
                "who can apply",
                "eligibility",
                "eligibility criteria",
            }:
                return 1.0

            if "who can apply" in section:
                return 1.0

            if "eligibility" in section:
                return 1.0

            if "who can apply" in text:
                return 0.35

            return 0.0

        # -----------------------------------------------------
        # Application
        # -----------------------------------------------------

        if intent == "application":
            if "online application" in section:
                return 1.0

            if section == "how to apply":
                return 1.0

            if "application" in section:
                return 0.90

            if "apply online" in text:
                return 0.60

            if "application form" in text:
                return 0.40

            return 0.0

        # -----------------------------------------------------
        # UPI activation
        # -----------------------------------------------------

        if intent == "upi_activation":
            if "upi" in section:
                if (
                    "activate" in section
                    or "activation" in section
                    or "register" in section
                    or "enable" in section
                ):
                    return 1.0

                return 0.90

            if "upi" in document:
                return 0.85

            if (
                "activate upi"
                in text
            ):
                return 0.80

            if "upi" in text:
                return 0.60

            return 0.0

        # -----------------------------------------------------
        # Petrol pump ATM
        # -----------------------------------------------------

        if intent == "petrol_pump_atm":
            section_or_doc = (
                f"{section} {document}"
            )

            has_petrol = (
                "petrol"
                in section_or_doc
            )

            has_pump = (
                "pump"
                in section_or_doc
            )

            has_atm = (
                "atm"
                in section_or_doc
                or "atm" in text
            )

            if (
                has_petrol
                and has_pump
                and has_atm
            ):
                return 1.0

            if (
                has_petrol
                and has_pump
            ):
                return 0.95

            if (
                "petrol pump"
                in text
                and has_atm
            ):
                return 0.90

            return 0.0

        # -----------------------------------------------------
        # Generic activation
        # -----------------------------------------------------

        if intent == "activation":
            if (
                "activation"
                in section
            ):
                return 1.0

            if "activate" in section:
                return 1.0

            if "registration" in section:
                return 0.85

            return 0.0

        # -----------------------------------------------------
        # Location
        # -----------------------------------------------------

        if intent == "location":
            if "location" in section:
                return 1.0

            if "atm" in section:
                return 0.95

            if (
                "location" in text
                or "atm" in text
            ):
                return 0.50

            return 0.0

        # -----------------------------------------------------
        # Definition
        # -----------------------------------------------------

        if intent == "definition":
            if (
                "what is" in combined
                or "definition" in combined
                or "meaning" in combined
            ):
                return 1.0

            return 0.0

        return 0.0

    # =========================================================
    # SECTION MATCH
    # =========================================================

    def section_match_score(
        self,
        query: str,
        result: dict[str, Any],
    ) -> float:

        intent = self.detect_intent(
            query
        )

        section = self.normalize_for_duplicate(
            str(
                result.get(
                    "section",
                    "",
                )
            )
        )

        document = self.normalize_for_duplicate(
            str(
                result.get(
                    "document_id",
                    "",
                )
            )
        )

        text = self.normalize_for_duplicate(
            str(
                result.get(
                    "text",
                    "",
                )
            )
        )

        if not section:
            return 0.0

        if section in {
            "na",
            "n a",
            "unknown",
            "none",
            "null",
        }:
            return 0.0

        # -----------------------------------------------------
        # Documents required
        # -----------------------------------------------------

        if intent == "documents_required":
            if section == "documents required":
                return 1.0

            if "documents required" in section:
                return 1.0

            if "required documents" in section:
                return 1.0

            return 0.0

        # -----------------------------------------------------
        # Eligibility
        # -----------------------------------------------------

        if intent == "eligibility":
            if section in {
                "who can apply",
                "eligibility",
                "eligibility criteria",
            }:
                return 1.0

            if "who can apply" in section:
                return 1.0

            if "eligibility" in section:
                return 1.0

            return 0.0

        # -----------------------------------------------------
        # Application
        # -----------------------------------------------------

        if intent == "application":
            if "online application" in section:
                return 1.0

            if section == "how to apply":
                return 1.0

            if "application" in section:
                return 0.90

            return 0.0

        # -----------------------------------------------------
        # UPI activation
        # -----------------------------------------------------

        if intent == "upi_activation":
            if (
                "upi" in section
                and (
                    "activate" in section
                    or "activation" in section
                    or "register" in section
                    or "enable" in section
                )
            ):
                return 1.0

            if "upi" in section:
                return 0.90

            if (
                "upi" in document
                and "upi" in text
            ):
                return 0.75

            return 0.0

        # -----------------------------------------------------
        # Petrol pump ATM
        # -----------------------------------------------------

        if intent == "petrol_pump_atm":
            combined = (
                f"{document} "
                f"{section} "
                f"{text}"
            )

            if (
                "petrol pump" in section
                and "atm" in combined
            ):
                return 1.0

            if (
                "petrol" in document
                and "pump" in document
                and "atm" in combined
            ):
                return 1.0

            if (
                "petrol pump" in text
                and "atm" in combined
            ):
                return 0.90

            return 0.0

        # -----------------------------------------------------
        # Activation
        # -----------------------------------------------------

        if intent == "activation":
            if "activate" in section:
                return 1.0

            if "activation" in section:
                return 1.0

            if "registration" in section:
                return 0.85

            return 0.0

        # -----------------------------------------------------
        # Location
        # -----------------------------------------------------

        if intent == "location":
            if "location" in section:
                return 1.0

            if "atm" in section:
                return 0.95

            return 0.0

        return 0.0

    # =========================================================
    # HYBRID SCORE
    # =========================================================

    def calculate_hybrid_score(
        self,
        query: str,
        result: dict[str, Any],
    ) -> float:

        rrf_score = result.get(
            "rrf_score",
            0.0,
        )

        metadata = self.metadata_score(
            query,
            result,
        )

        intent = self.intent_score(
            query,
            result,
        )

        entity = self.entity_score(
            query,
            result,
        )

        section_match = (
            self.section_match_score(
                query,
                result,
            )
        )

        phrase = (
            self.phrase_match_score(
                query,
                result,
            )
        )

        intent_candidate = float(
            result.get(
                "intent_candidate_score",
                0.0,
            )
        )

        # Normalize intent candidate contribution.
        intent_candidate_normalized = min(
            intent_candidate / 20.0,
            1.0,
        )

        result[
            "metadata_score"
        ] = metadata

        result[
            "intent_score"
        ] = intent

        result[
            "entity_score"
        ] = entity

        result[
            "section_match_score"
        ] = section_match

        result[
            "phrase_match_score"
        ] = phrase

        result[
            "intent_candidate_score_normalized"
        ] = intent_candidate_normalized

        final_score = (
            rrf_score
            + 0.08 * metadata
            + 0.10 * intent
            + self.SECTION_MATCH_BOOST
            * section_match
            + self.ENTITY_MATCH_BOOST
            * entity
            + self.PHRASE_MATCH_BOOST
            * phrase
            + 0.20
            * intent_candidate_normalized
        )

        result[
            "hybrid_score"
        ] = final_score

        return final_score

    # =========================================================
    # DUPLICATES
    # =========================================================

    def remove_exact_duplicates(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        seen = set()
        unique_results = []

        for result in results:
            key = self.get_chunk_key(
                result
            )

            if key in seen:
                continue

            seen.add(key)
            unique_results.append(
                result
            )

        return unique_results

    def remove_near_duplicates(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        unique_results = []

        for result in results:
            current_text = (
                self.normalize_for_duplicate(
                    result.get(
                        "text",
                        "",
                    )
                )
            )

            if not current_text:
                continue

            is_duplicate = False

            for existing in unique_results:
                existing_text = (
                    self.normalize_for_duplicate(
                        existing.get(
                            "text",
                            "",
                        )
                    )
                )

                if not existing_text:
                    continue

                similarity = (
                    SequenceMatcher(
                        None,
                        current_text,
                        existing_text,
                    ).ratio()
                )

                if (
                    similarity
                    >= self.NEAR_DUPLICATE_THRESHOLD
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(
                    result
                )

        return unique_results

    # =========================================================
    # DOCUMENT / SECTION KEYS
    # =========================================================

    def get_document_key(
        self,
        result: dict[str, Any],
    ) -> str:

        document_id = result.get(
            "document_id",
            "",
        )

        if not document_id:
            metadata = result.get(
                "metadata",
                {},
            )

            document_id = metadata.get(
                "document_id",
                "",
            )

        if not document_id:
            source = result.get(
                "source",
                "",
            )

            if source:
                document_id = Path(
                    source
                ).stem

        return self.normalize_for_duplicate(
            str(document_id)
        )

    def get_section_key(
        self,
        result: dict[str, Any],
    ) -> str:

        section = result.get(
            "section",
            "",
        )

        if not section:
            metadata = result.get(
                "metadata",
                {},
            )

            section = metadata.get(
                "section",
                "",
            )

        return self.normalize_for_duplicate(
            str(section)
        )

    # =========================================================
    # DOCUMENT DIVERSITY
    # =========================================================

    def remove_document_duplicates(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        selected = []

        document_counts: dict[str, int] = {}
        seen_document_sections = set()

        for result in results:
            document_key = (
                self.get_document_key(
                    result
                )
            )

            section_key = (
                self.get_section_key(
                    result
                )
            )

            if not document_key:
                selected.append(result)

                if len(selected) >= self.top_k:
                    break

                continue

            unknown_section = (
                not section_key
                or section_key
                in {
                    "na",
                    "n a",
                    "unknown",
                    "none",
                    "null",
                }
            )

            section_identity = (
                document_key,
                section_key,
            )

            if (
                not unknown_section
                and section_identity
                in seen_document_sections
            ):
                continue

            if unknown_section:
                if (
                    document_key
                    in document_counts
                ):
                    continue

            current_count = (
                document_counts.get(
                    document_key,
                    0,
                )
            )

            if (
                current_count
                >= self.MAX_RESULTS_PER_DOCUMENT
            ):
                continue

            selected.append(result)

            document_counts[
                document_key
            ] = current_count + 1

            if not unknown_section:
                seen_document_sections.add(
                    section_identity
                )

            if len(selected) >= self.top_k:
                break

        return selected

    # =========================================================
    # RERANK
    # =========================================================

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        if not results:
            return []

        pairs = []

        for result in results:
            document_id = result.get(
                "document_id",
                "",
            )

            section = result.get(
                "section",
                "N/A",
            )

            text = result.get(
                "text",
                "",
            )

            passage = (
                f"Document: {document_id}\n"
                f"Section: {section}\n"
                f"Content: {text}"
            )

            pairs.append(
                (
                    query,
                    passage,
                )
            )

        scores = self.reranker.predict(
            pairs
        )

        for result, score in zip(
            results,
            scores,
        ):
            result[
                "reranker_score"
            ] = float(score)

        results.sort(
            key=lambda item: item.get(
                "reranker_score",
                0.0,
            ),
            reverse=True,
        )

        return results

    # =========================================================
    # FINAL RANKING
    # =========================================================

    @staticmethod
    def normalize_reranker_score(
        score: float,
    ) -> float:

        score = max(
            min(score, 50.0),
            -50.0,
        )

        return 1.0 / (
            1.0 + math.exp(-score)
        )

    def calculate_final_ranking_score(
        self,
        result: dict[str, Any],
    ) -> float:

        reranker_score = float(
            result.get(
                "reranker_score",
                0.0,
            )
        )

        normalized_reranker = (
            self.normalize_reranker_score(
                reranker_score
            )
        )

        intent = float(
            result.get(
                "intent_score",
                0.0,
            )
        )

        section_match = float(
            result.get(
                "section_match_score",
                0.0,
            )
        )

        metadata = float(
            result.get(
                "metadata_score",
                0.0,
            )
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # An exact intent/section match must be able
        # to beat a generic reranker match.
        # -----------------------------------------------------

        final_score = (
            self.RERANKER_WEIGHT
            * normalized_reranker
            + self.INTENT_WEIGHT
            * intent
            + self.SECTION_WEIGHT
            * section_match
            + self.METADATA_WEIGHT
            * metadata
        )

        result[
            "normalized_reranker"
        ] = normalized_reranker

        result[
            "final_ranking_score"
        ] = final_score

        return final_score

    # =========================================================
    # INTENT PRIORITY
    # =========================================================

    def apply_intent_priority(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Final deterministic correction for highly specific
        enterprise queries.

        This is NOT a hard-coded answer.

        It only prioritizes a chunk when its document/section
        actually contains the terminology requested by the user.
        """

        intent = self.detect_intent(
            query
        )

        for result in results:
            result[
                "intent_priority"
            ] = 0.0

            section = str(
                result.get(
                    "section",
                    "",
                )
            ).lower()

            document = str(
                result.get(
                    "document_id",
                    "",
                )
            ).lower()

            text = str(
                result.get(
                    "text",
                    "",
                )
            ).lower()

            combined = (
                f"{document} "
                f"{section} "
                f"{text}"
            )

            priority = 0.0

            if intent == "documents_required":
                if (
                    "documents required"
                    in section
                ):
                    priority = 1.0
                elif (
                    "required documents"
                    in section
                ):
                    priority = 1.0

            elif intent == "eligibility":
                if (
                    "who can apply"
                    in section
                    or "eligibility"
                    in section
                ):
                    priority = 1.0

            elif intent == "application":
                if (
                    "online application"
                    in section
                ):
                    priority = 1.0
                elif (
                    "how to apply"
                    in section
                ):
                    priority = 1.0

            elif intent == "upi_activation":
                if (
                    "upi" in combined
                    and (
                        "activate"
                        in combined
                        or "activation"
                        in combined
                        or "register"
                        in combined
                        or "enable"
                        in combined
                    )
                ):
                    priority = 1.0
                elif "upi" in section:
                    priority = 0.90

            elif intent == "petrol_pump_atm":
                if (
                    "petrol pump"
                    in combined
                    and "atm"
                    in combined
                ):
                    priority = 1.0
                elif (
                    "petrol" in combined
                    and "pump" in combined
                    and "atm" in combined
                ):
                    priority = 1.0

            result[
                "intent_priority"
            ] = priority

            if priority > 0:
                result[
                    "final_ranking_score"
                ] += (
                    0.25 * priority
                )

        results.sort(
            key=lambda item: item.get(
                "final_ranking_score",
                0.0,
            ),
            reverse=True,
        )

        return results

    # =========================================================
    # MAIN SEARCH
    # =========================================================

    def search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        normalized_query = (
            self.normalize_query(query)
        )

        # -----------------------------------------------------
        # 1. Semantic retrieval
        # -----------------------------------------------------

        semantic_results = (
            self.semantic_search(
                normalized_query
            )
        )

        # -----------------------------------------------------
        # 2. BM25 retrieval
        # -----------------------------------------------------

        keyword_results = (
            self.keyword_search(
                normalized_query
            )
        )

        # -----------------------------------------------------
        # 3. Intent-aware retrieval
        # -----------------------------------------------------

        intent_results = (
            self.intent_candidate_search(
                normalized_query
            )
        )

        # -----------------------------------------------------
        # 4. Merge ALL retrieval channels
        # -----------------------------------------------------

        candidates = (
            self.merge_candidates(
                semantic_results,
                keyword_results,
                intent_results,
            )
        )

        # -----------------------------------------------------
        # 5. RRF
        # -----------------------------------------------------

        candidates = (
            self.reciprocal_rank_fusion(
                candidates
            )
        )

        # -----------------------------------------------------
        # 6. Hybrid scoring
        # -----------------------------------------------------

        for result in candidates:
            self.calculate_hybrid_score(
                normalized_query,
                result,
            )

        candidates.sort(
            key=lambda item: item.get(
                "hybrid_score",
                0.0,
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # 7. Exact duplicates
        # -----------------------------------------------------

        candidates = (
            self.remove_exact_duplicates(
                candidates
            )
        )

        # -----------------------------------------------------
        # 8. Near duplicates
        # -----------------------------------------------------

        candidates = (
            self.remove_near_duplicates(
                candidates
            )
        )

        # -----------------------------------------------------
        # 9. Rerank
        # -----------------------------------------------------

        rerank_candidates = candidates[
            : max(
                self.candidate_k,
                self.INTENT_CANDIDATE_LIMIT,
            )
        ]

        reranked = self.rerank(
            normalized_query,
            rerank_candidates,
        )

        # -----------------------------------------------------
        # 10. Reranker threshold
        # -----------------------------------------------------

        relevant_results = []

        for result in reranked:
            score = result.get(
                "reranker_score",
                0.0,
            )

            if (
                score
                >= self.RERANKER_THRESHOLD
            ):
                relevant_results.append(
                    result
                )

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        if not relevant_results:
            if reranked:
                relevant_results = [
                    reranked[0]
                ]
            else:
                return []

        # -----------------------------------------------------
        # 11. Exact duplicate filter
        # -----------------------------------------------------

        relevant_results = (
            self.remove_exact_duplicates(
                relevant_results
            )
        )

        # -----------------------------------------------------
        # 12. Near duplicate filter
        # -----------------------------------------------------

        relevant_results = (
            self.remove_near_duplicates(
                relevant_results
            )
        )

        # -----------------------------------------------------
        # 13. Final ranking
        # -----------------------------------------------------

        for result in relevant_results:
            self.calculate_final_ranking_score(
                result
            )

        # -----------------------------------------------------
        # 14. Intent priority
        # -----------------------------------------------------

        relevant_results = (
            self.apply_intent_priority(
                normalized_query,
                relevant_results,
            )
        )

        # -----------------------------------------------------
        # 15. Document diversity
        # -----------------------------------------------------

        relevant_results = (
            self.remove_document_duplicates(
                relevant_results
            )
        )

        # -----------------------------------------------------
        # 16. Final sort
        # -----------------------------------------------------

        relevant_results.sort(
            key=lambda item: item.get(
                "final_ranking_score",
                0.0,
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # 17. Final Top-K
        # -----------------------------------------------------

        return relevant_results[
            : self.top_k
        ]

    # =========================================================
    # DEBUG OUTPUT
    # =========================================================

    def print_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> None:

        print()
        print("=" * 70)
        print(
            f"QUERY: {query}"
        )
        print("=" * 70)

        print(
            f"Detected intent: "
            f"{self.detect_intent(query)}"
        )

        if not results:
            print(
                "No relevant results found."
            )
            return

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print()
            print("-" * 35)

            print(
                f"Rank: {rank}"
            )

            print(
                "Final ranking score: "
                f"{result.get('final_ranking_score', 0.0):.4f}"
            )

            print(
                "Reranker score: "
                f"{result.get('reranker_score', 0.0):.4f}"
            )

            print(
                "Normalized reranker: "
                f"{result.get('normalized_reranker', 0.0):.4f}"
            )

            print(
                "Hybrid score: "
                f"{result.get('hybrid_score', 0.0):.4f}"
            )

            print(
                "RRF score: "
                f"{result.get('rrf_score', 0.0):.4f}"
            )

            print(
                "Semantic score: "
                f"{result.get('semantic_score', 0.0):.4f}"
            )

            print(
                "Keyword score: "
                f"{result.get('bm25_score', 0.0):.4f}"
            )

            print(
                "Intent candidate score: "
                f"{result.get('intent_candidate_score', 0.0):.4f}"
            )

            print(
                "Entity score: "
                f"{result.get('entity_score', 0.0):.4f}"
            )

            print(
                "Phrase match score: "
                f"{result.get('phrase_match_score', 0.0):.4f}"
            )

            print(
                "Metadata score: "
                f"{result.get('metadata_score', 0.0):.4f}"
            )

            print(
                "Intent score: "
                f"{result.get('intent_score', 0.0):.4f}"
            )

            print(
                "Section match score: "
                f"{result.get('section_match_score', 0.0):.4f}"
            )

            print(
                "Intent priority: "
                f"{result.get('intent_priority', 0.0):.4f}"
            )

            print(
                "Document: "
                f"{result.get('document_id', 'N/A')}"
            )

            print(
                "Section: "
                f"{result.get('section', 'N/A')}"
            )

            print(
                "Chunk ID: "
                f"{result.get('chunk_id', 'N/A')}"
            )

            print()
            print("Text:")

            print(
                result.get(
                    "text",
                    "",
                )
            )

        print()
        print("=" * 70)


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    retriever = HybridRetriever(
        top_k=5
    )

    queries = [
        "What documents are required for an Agriculture Loan?",
        "Who can apply for an Agriculture Loan?",
        "How can I apply for an Agriculture Loan online?",
        "How do I activate UPI with ZX Bank?",
        "Where can I find ZX Bank ATM locations at major petrol pumps?",
    ]

    for query in queries:

        results = retriever.search(
            query
        )

        retriever.print_results(
            query,
            results,
        )