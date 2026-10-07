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