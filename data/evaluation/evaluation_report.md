# Enterprise RAG Knowledge Assistant — Evaluation Report

## Dataset

- Dataset: ZX Bank
- Documents: 355
- Total chunks: 3,465
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Vector store: FAISS
- LLM: Groq — openai/gpt-oss-20b

## Retrieval Evaluation

| Metric | Result |
|---|---:|
| Questions | 8 |
| Hit@1 | 100.00% |
| Hit@5 | 100.00% |
| MRR | 1.000 |

## Answer Evaluation

| Metric | Result |
|---|---:|
| Questions | 5 |
| Keyword Coverage | 96.67% |

The answer evaluation intentionally reports 96.67%. The online-application case did not receive full keyword coverage because the evaluator expected the exact phrase "upload documents", while the generated answer used equivalent wording.

## Hallucination Evaluation

- Test: Agriculture Loan interest-rate question
- Result: PASS
- The system correctly refused to invent a numerical interest rate when the retrieved documents did not contain one.

## Reliability Evaluation

| Metric | Result |
|---|---:|
| Tests | 5 |
| Passed | 5 |
| Failed | 0 |

Reliability tests covered empty questions, out-of-domain questions, unavailable information, complex queries, and normal factual questions.

## API Regression Testing

| Metric | Result |
|---|---:|
| Tests | 7 |
| Passed | 7 |
| Failed | 0 |

API tests covered:

- Health endpoint
- Normal question request
- Empty question validation
- Whitespace question validation
- Maximum question length validation
- Conversation history limit
- CORS preflight

## Overall Status

The Enterprise RAG Knowledge Assistant passed retrieval, hallucination, reliability, and API regression testing.

The current evaluation demonstrates:

- 100% Hit@1 retrieval accuracy
- 100% Hit@5 retrieval accuracy
- 1.000 MRR
- 96.67% answer keyword coverage
- 5/5 reliability tests passed
- 7/7 API regression tests passed
- Hallucination test passed
