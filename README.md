# DocTalk — Chat with Your Documents

I built this to learn how RAG (Retrieval-Augmented Generation) works in practice. 
The idea is simple — upload any document and ask questions about it. The app finds 
the most relevant parts of the document and uses an LLM to answer based strictly 
on what's there.

No hallucination — if the answer isn't in the document, it says so.

---

## How it works

When you upload a document, it gets split into small chunks and converted into 
vectors using a HuggingFace embedding model. Those vectors are stored in a FAISS 
index. When you ask a question, the same embedding model converts your question 
into a vector, finds the closest matching chunks, and passes them to Llama 3 
(via Groq) to generate an answer grounded in the document.

Each document gets its own FAISS index so multiple documents can be stored and 
queried independently. You can also merge indexes to query across multiple documents 
at once.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) lets an LLM answer questions grounded in a 
specific document rather than relying on its training data alone.

1. **Index** — Split the document into chunks and embed them as vectors
2. **Retrieve** — Find the chunks most relevant to the user's question
3. **Generate** — Pass those chunks to an LLM to produce a grounded answer

---

## Stack

| Component | Tool |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector Store | FAISS |
| LLM | Groq API — Llama 3.3 70b (free tier) |
| Orchestration | LangChain |
| UI | Streamlit (coming in Week 4) |
| Notebooks | Google Colab |

Total API cost: $0 — all free tools.

---

## Supported file types

PDF, DOCX, TXT — with validation to catch empty or unreadable files before indexing.

---

## Project structure

```
doctalk/
├── notebooks/
│   ├── Week1_Foundations.ipynb
│   ├── Week2_LLM_Integration.ipynb
│   └── Week3_Refinements.ipynb
├── Results/
│   ├── week1_output.pdf
│   ├── week2_output.pdf
│   └── week3_output.pdf
├── LEARNINGS.md
├── requirements.txt
└── README.md

```

## Build log

- [x] Week 1 — PDF loading, chunking, HuggingFace embeddings, FAISS index
- [x] Week 2 — Groq LLM integration, prompt engineering, answer grounding
- [x] Week 3 — Multi file type support, per-document indexing, multi-doc querying, edge cases
- [ ] Week 4 — Folder indexing, table extraction, Streamlit UI
- [ ] Week 5 — Reranker, raw pipeline rewrite, embedding experiments

---

## Things I learned along the way

The full writeup is in [LEARNINGS.md](LEARNINGS.md) but the short version:

Chunk size matters more than I expected. Switching from 500 to 200 characters per 
chunk was the single biggest improvement to retrieval quality. A dense chunk with 
5 facts scores mediocrely for all 5 queries — a focused chunk with 1 fact scores 
precisely for that query.

Compression sounded smart but introduced hallucination. The simpler pipeline 
outperformed the complex one. Fix the data first, add complexity later.

FAISS has a built-in merge_from method that makes combining multiple indexes 
trivial — one line to merge any number of indexes for cross-document querying.

---

## Run it yourself

Open any notebook in the `notebooks/` folder directly in Google Colab — no local 
setup needed. You'll need a free API key from [console.groq.com](https://console.groq.com).

---

## About

Built as a self-learning project to break into AI/ML engineering.
Follow the build on GitHub — new notebooks added weekly!
