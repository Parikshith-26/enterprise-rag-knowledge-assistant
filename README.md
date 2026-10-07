# Enterprise RAG Knowledge Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) knowledge assistant that combines document ingestion, semantic search, hybrid retrieval, reranking, conversational memory, grounded generation, source citations, evaluation, API serving, and containerized deployment.

The current validated implementation uses the **ZX Bank multi-format document corpus** as its reference knowledge base. The architecture is designed to evolve into a document-upload-driven RAG assistant where users can upload their own enterprise documents and query them conversationally.

---

## 🚀 Project Overview

Traditional keyword-based enterprise search often struggles to understand the intent behind natural-language questions.

This project implements a complete RAG pipeline that:

1. Ingests enterprise documents
2. Cleans and normalizes content
3. Splits documents into meaningful chunks
4. Generates semantic embeddings
5. Stores embeddings in FAISS
6. Performs hybrid semantic + keyword retrieval
7. Applies intent-aware reranking
8. Generates grounded answers using an LLM
9. Returns supporting source information
10. Maintains conversational context
11. Evaluates retrieval and answer quality
12. Exposes the system through FastAPI
13. Provides a Streamlit user interface
14. Runs through Docker and Docker Compose

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      User            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Streamlit UI     │
                         │       Port 8501      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │       Port 8000      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      RAG Answer Pipeline      │
                    └──────────────┬────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          Question Rewrite   Hybrid Retrieval   Conversation
                 │                 │              Context
                 │          ┌──────┴──────┐
                 │          │             │
                 │       Semantic       BM25
                 │       Retrieval     Retrieval
                 │          │             │
                 │          └──────┬──────┘
                 │                 │
                 │                 ▼
                 │          Intent-aware
                 │            Reranking
                 │                 │
                 └─────────────────┤
                                   ▼
                           Retrieved Context
                                   │
                                   ▼
                           ┌───────────────┐
                           │   Groq LLM    │
                           └───────┬───────┘
                                   │
                                   ▼
                         Answer + Sources


📚 Current Knowledge Base
The current implementation is validated using the ZX Bank corpus.
Dataset statistics
Metric	Value
Documents	355
Total chunks	3,465
Document formats	PDF, DOCX, HTML, Markdown, PPTX
Embedding dimensions	384
Vector store	FAISS


The ingestion pipeline preserves useful document structure such as headings, sections, pages, and slides.
🔄 RAG Pipeline
1. Document Ingestion
The system supports:
- PDF
- DOCX
- HTML
- Markdown
- PPTX
Each document is converted into a common internal document representation.
2. Text Cleaning
The cleaning pipeline:
- normalizes line endings
- removes unnecessary whitespace
- normalizes blank lines
- preserves meaningful document structure
3. Intelligent Chunking
Documents are divided into retrieval-friendly chunks while preserving:
- document title
- section headings
- page information
- slide information
- document metadata
- document context
The current ZX Bank corpus produces:
355 documents
↓
3,465 chunks

4. Embeddings
Semantic embeddings are generated using:
sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
384

5. Vector Search
FAISS is used for efficient similarity search.
The current implementation uses:
FAISS IndexFlatIP

with normalized embeddings.
6. Hybrid Retrieval
The retrieval pipeline combines:
- semantic similarity
- BM25 keyword retrieval
- intent-aware candidate selection
- metadata matching
- section matching
- phrase/entity matching
This improves retrieval for both conceptual and exact-match queries.
7. Reranking
Retrieved candidates are reranked using a combination of:
- semantic relevance
- intent relevance
- section relevance
- metadata relevance
- phrase/entity matching
The system also reduces duplicate results from the same document.
8. Conversational Query Rewriting
Follow-up questions are rewritten into standalone search queries.
Example:
User:
What is an Agriculture Loan?

User:
How can I apply for it online?

The second question can be rewritten into:
How to apply for an Agriculture Loan online

This allows retrieval to use the correct context.
9. Grounded Answer Generation
The Groq LLM generates answers using retrieved context.
The generation pipeline is explicitly instructed to:
- use retrieved information only
- avoid inventing information
- state when information is unavailable
- preserve relevant procedural details
- provide concise answers
- combine consistent supporting sources
🤖 LLM
Current model:
openai/gpt-oss-20b

Provider:
Groq

The API key is loaded through environment variables and is never stored in the repository.
📌 Source Citations
Every generated answer can return supporting source metadata including:
- document ID
- document title
- section
- chunk ID
- document type
- source
This allows users to understand where the answer originated.
🧠 Conversational Memory
The assistant supports multi-turn conversations.
Example:
User:
What is an Agriculture Loan?

Assistant:
...

User:
How can I apply for it online?

Assistant:
...

The system uses conversation history to resolve references such as:
it
this
that loan
the above process

before performing retrieval.
🛡️ Hallucination Handling
The system is designed to avoid generating unsupported facts.
For example, when asked for an exact Agriculture Loan interest rate that is not present in the retrieved documents, the assistant responds that the information is unavailable rather than inventing a numerical value.
📊 Evaluation
The current system has been evaluated across retrieval, answer quality, hallucination resistance, reliability, and API behavior.
Retrieval Evaluation
Metric	Result
Questions	8
Hit@1	100.00%
Hit@5	100.00%
MRR	1.000


Answer Evaluation
Metric	Result
Questions	5
Keyword Coverage	96.67%


The 96.67% score is intentionally retained. One online-application case did not receive full keyword coverage because the evaluator expected the exact phrase upload documents, while the generated answer used equivalent wording.
Reliability Testing
Metric	Result
Tests	5
Passed	5
Failed	0


Tests cover:
- empty questions
- out-of-domain questions
- unavailable information
- complex questions
- normal factual questions
API Regression Testing
Metric	Result
Tests	7
Passed	7
Failed	0


API tests cover:
- health endpoint
- normal question requests
- empty question validation
- whitespace validation
- maximum question length
- conversation history limits
- CORS preflight
🌐 API
FastAPI provides the backend service.
Health
GET /health

Ask a Question
POST /ask

Example request:
{
  "question": "What documents are required for an Agriculture Loan?",
  "conversation_history": []
}

Example response structure:
{
  "question": "What documents are required for an Agriculture Loan?",
  "search_question": "documents required for agriculture loan",
  "answer": "...",
  "sources": []
}

Interactive API documentation is available through:
http://localhost:8000/docs

🖥️ Streamlit Application
The Streamlit frontend provides:
- conversational chat interface
- question answering
- conversation history
- source display
- API health handling
- error handling
Application:
http://localhost:8501

🐳 Docker
The application is containerized using Docker.
Architecture:
Docker Compose
│
├── API Container
│   └── FastAPI + RAG Pipeline
│
└── Streamlit Container
    └── Frontend

Start the application
Make sure Docker Desktop is running.
docker compose up -d

Check containers:
docker compose ps

Open:
http://localhost:8501

Stop the application
docker compose down

💻 Local Development
Create and activate the virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run FastAPI:
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Run Streamlit in another terminal:
streamlit run app.py

For Docker Compose, Streamlit communicates with the API using the Docker service name api. For direct local execution, the API URL should point to 127.0.0.1:8000.

📁 Project Structure
enterprise-rag/
│
├── app.py
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── .env
├── .gitignore
├── .dockerignore
│
├── data/
│   ├── documents/
│   ├── processed/
│   ├── indexes/
│   ├── evaluation/
│   └── logs/
│
└── src/
    ├── ingestion/
    ├── chunking/
    ├── embeddings/
    ├── vectorstore/
    ├── retrieval/
    ├── generation/
    ├── evaluation/
    ├── api/
    ├── ingest.py
    ├── logging_config.py
    └── test_*.py

🔐 Security
The project includes several security-oriented practices:
- API keys loaded through environment variables
- .env excluded from Git
- Docker secrets not hardcoded into source code
- API request validation
- question length limits
- conversation history limits
- restricted CORS origins
- internal errors are not exposed to users
- application logging
🧪 Testing
Individual test modules cover:
Document loaders
Chunking
Embeddings
Retrieval
Hybrid retrieval
Answer generation
Hallucination
Reliability
API

Run the API regression tests:
python src/test_api.py

Expected result:
Passed: 7/7
Failed: 0/7
ALL API TESTS PASSED

🔮 Planned Evolution: Upload-Driven RAG
The current implementation uses the ZX Bank corpus as the validated reference dataset.
The next evolution of the system is to make the knowledge base document-driven rather than dataset-driven.
The planned workflow is:
User uploads documents
        ↓
Document validation
        ↓
Multi-format ingestion
        ↓
Cleaning
        ↓
Chunking
        ↓
Embedding generation
        ↓
FAISS index creation
        ↓
User asks questions
        ↓
Hybrid retrieval
        ↓
Reranking
        ↓
Grounded answer
        ↓
Sources

This would allow users to upload their own enterprise documents and create a temporary/private knowledge base without requiring a preconfigured dataset.
Supported upload formats are planned to include:
- PDF
- DOCX
- PPTX
- Markdown
- HTML
- TXT
🚀 Future Improvements
- User document upload interface
- Dynamic document indexing
- Per-session knowledge bases
- Multiple independent document collections
- Persistent vector-store management
- Better citation rendering
- Advanced evaluation datasets
- Authentication and authorization
- Document-level access control
- Production database integration
- Observability dashboards
- Cloud deployment
- Automated CI/CD
- Advanced reranking models
- Streaming responses
🎯 Key Skills Demonstrated
This project demonstrates practical experience with:
- Retrieval-Augmented Generation
- NLP
- Semantic Search
- Vector Databases
- FAISS
- BM25
- Hybrid Retrieval
- Reranking
- Sentence Transformers
- LLM Integration
- Prompt Engineering
- Conversational AI
- Hallucination Control
- FastAPI
- Streamlit
- Docker
- REST APIs
- Python
- Evaluation & Testing
- Application Logging
- AI System Architecture
👨‍💻 Project Status
Current
✅ Multi-format ingestion
✅ Document cleaning
✅ Intelligent chunking
✅ Embeddings
✅ FAISS vector search
✅ Hybrid retrieval
✅ Intent-aware reranking
✅ Groq LLM generation
✅ Source citations
✅ Conversational memory
✅ FastAPI
✅ Streamlit
✅ Docker
✅ API validation
✅ CORS
✅ Logging
✅ Evaluation
✅ Reliability testing
✅ Hallucination testing
