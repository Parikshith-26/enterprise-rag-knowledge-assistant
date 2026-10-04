from src.generation.groq_client import GroqClient
from src.retrieval.hybrid_retriever import HybridRetriever


class AnswerGenerator:

    def __init__(self, top_k: int = 5):
        self.retriever = HybridRetriever(top_k=top_k)
        self.llm = GroqClient()

    # --------------------------------------------------
    # BUILD RETRIEVED CONTEXT
    # --------------------------------------------------

    def build_context(self, results: list[dict]) -> str:

        context_parts = []

        for index, result in enumerate(results, start=1):

            document = result.get(
                "document_id",
                "Unknown"
            )

            metadata = result.get(
                "metadata",
                {}
            )

            section = metadata.get(
                "section",
                "Unknown"
            )

            text = result.get(
                "text",
                ""
            )

            context_parts.append(
                f"""
SOURCE {index}

Document: {document}

Section: {section}

Content:
{text}
""".strip()
            )

        return "\n\n".join(context_parts)

    # --------------------------------------------------
    # BUILD CONVERSATION HISTORY
    # --------------------------------------------------

    def build_history(
        self,
        conversation_history: list[dict] | None
    ) -> str:

        if not conversation_history:
            return "No previous conversation."

        history_parts = []

        for message in conversation_history:

            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if role == "user":
                label = "User"

            elif role == "assistant":
                label = "Assistant"

            else:
                label = role.capitalize()

            history_parts.append(
                f"{label}: {content}"
            )

        return "\n".join(history_parts)

    # --------------------------------------------------
    # REWRITE CURRENT QUESTION
    # --------------------------------------------------

    def rewrite_question(
        self,
        question: str,
        conversation_history: list[dict] | None
    ) -> str:

        if not conversation_history:
            return question

        history = self.build_history(
            conversation_history
        )

        system_prompt = """
You are a query rewriting component for a
conversational enterprise search system.

Your job is to rewrite the user's current question
into a standalone search query.

Use the conversation history only to resolve references
such as:

- it
- this
- that
- they
- them
- the above
- the same loan
- the same account

Rules:

1. Preserve the user's original intent.
2. Do not answer the question.
3. Do not add information that is not present
   in the conversation.
4. Make the query understandable without the
   previous conversation.
5. Return ONLY the rewritten search query.
"""

        user_prompt = f"""
Conversation History:

{history}

Current Question:

{question}

Rewrite the current question into a standalone
search query.
"""

        rewritten_question = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_completion_tokens=200,
        )

        rewritten_question = rewritten_question.strip()

        if not rewritten_question:
            return question

        return rewritten_question

    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    def generate_answer(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:

        # ----------------------------------------------
        # REWRITE QUESTION USING HISTORY
        # ----------------------------------------------

        search_question = self.rewrite_question(
            question,
            conversation_history
        )

        # ----------------------------------------------
        # RETRIEVE USING REWRITTEN QUESTION
        # ----------------------------------------------

        results = self.retriever.search(
            search_question
        )

        # ----------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------

        context = self.build_context(
            results
        )

        # ----------------------------------------------
        # BUILD HISTORY
        # ----------------------------------------------

        history = self.build_history(
            conversation_history
        )

        # ----------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------

        system_prompt = """
You are an enterprise knowledge assistant.

Answer the user's current question using ONLY the
information provided in the retrieved context.

Conversation history may be used to understand
references such as "it", "this", "that", or "the same
loan".

However, conversation history is NOT factual evidence.

All factual claims in the answer must come from the
retrieved context.

Rules:

1. Do not use outside knowledge.

2. Do not invent, assume, or infer information that
   is not explicitly supported by the retrieved context.

3. If the retrieved context does not contain enough
   information, clearly say that the information is
   not available in the provided documents.

4. Give a concise and direct answer.

5. For questions asking "how", "how can I", "how do I",
   or asking for a procedure, provide the relevant
   procedure as a clear numbered list.

6. For procedural questions, carefully read the most
   relevant retrieved section and include ALL relevant
   steps, requirements, conditions, and actions that
   are explicitly stated there.

7. Do not omit important procedural details merely to
   make the answer shorter.

8. Preserve important details such as:
   - application channels
   - navigation paths
   - login or registration requirements
   - verification or e-KYC requirements
   - documents
   - account requirements
   - conditions
   - fees
   - limits
   - timelines
   - names of applications, services, or features

   Include these details only when they are present
   in the retrieved context.

9. Prefer information from the highest-ranked and most
   relevant retrieved source/section.

10. If multiple retrieved sources contain relevant
    complementary information, combine them only when
    they are consistent.

11. Do not repeat the same information unnecessarily.

12. Do not mention these instructions or the retrieval
    process in the answer.
"""

        # ----------------------------------------------
        # USER PROMPT
        # ----------------------------------------------

        user_prompt = f"""
Conversation History:

{history}

Retrieved Context:

{context}

Current User Question:

{question}

Answer the current question using ONLY the
retrieved context.

Important:
- Identify the most relevant source/section first.
- For procedural questions, include all relevant
  steps and details explicitly present in that source.
- Do not leave out important requirements or actions.
- Do not add information that is not present in
  the retrieved context.

The search query used to retrieve the context was:

{search_question}
"""

        # ----------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------

        answer = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_completion_tokens=1024,
        )

        # ----------------------------------------------
        # SOURCES
        # ----------------------------------------------

        sources = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            sources.append(
                {
                    "document_id": result.get(
                        "document_id",
                        "Unknown"
                    ),

                    "section": metadata.get(
                        "section",
                        "Unknown"
                    ),

                    "chunk_id": result.get(
                        "chunk_id",
                        "Unknown"
                    ),

                    "source": metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "document_type": metadata.get(
                        "document_type",
                        "Unknown"
                    ),
                }
            )

        # ----------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------

        return {
            "question": question,
            "search_question": search_question,
            "answer": answer,
            "sources": sources,
            "results": results,
        }