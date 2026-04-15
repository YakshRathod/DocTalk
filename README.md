# 📄 DocTalk — RAG-Powered Document Q&A App

Upload any document and chat with it. Built with LangChain, HuggingFace Embeddings, FAISS, Groq (Llama 3), and Streamlit.

> 🚧 **Work in progress** — built as a personal learning project to explore RAG (Retrieval-Augmented Generation).

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that lets an LLM answer questions grounded in a specific document rather than relying on its training data alone.

1. **Index** — Split the document into chunks and embed them as vectors
2. **Retrieve** — Find the chunks most relevant to the user's question
3. **Generate** — Pass those chunks to an LLM to produce a grounded answer

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector Store | FAISS |
| LLM | Groq API — Llama 3 (free tier) |
| Orchestration | LangChain |
| UI | Streamlit |
| Notebooks | Google Colab |

**Total API cost: $0** — all free tools.

---

## 📁 Project Structure

```
doctalk/
├── notebooks/
│   ├── Week1_Foundations.ipynb     # PDF loading, chunking, embeddings, FAISS
│   ├── Week2_LLM_Pipeline.ipynb    # Adding Groq LLM for answer generation
│   ├── Week3_Refinements.ipynb     # Edge cases, multi-file support
│   └── Week4_Streamlit_App.ipynb   # Building and deploying the UI
├── app/
│   └── app.py                      # Final Streamlit app
├── docs/
│   └── architecture.png            # System diagram
├── requirements.txt
└── README.md
```

---

## 📅 Build Log

- [x] **Week 1** — PDF loading, chunking, HuggingFace embeddings, FAISS index
- [ ] **Week 2** — Groq LLM integration, prompt engineering, answer generation
- [ ] **Week 3** — Multi-file support, source citation, edge case handling
- [ ] **Week 4** — Streamlit UI, deployment on Streamlit Cloud

---

## 🚀 Run It Yourself

### Option 1: Google Colab (easiest)
Open any notebook in the `notebooks/` folder directly in Colab — no local setup needed.

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/doctalk.git
cd doctalk
pip install -r requirements.txt
streamlit run app/app.py
```

You'll need a free [Groq API key](https://console.groq.com).

---

## 📦 Requirements

```
langchain
langchain-community
langchain-huggingface
faiss-cpu
pymupdf
sentence-transformers
groq
streamlit
```

---

## 💡 Learnings & Challenges

*(Updated weekly as the project progresses)*

- Chunk size significantly affects retrieval quality — smaller chunks are more precise but miss broader context
- HuggingFace embeddings are surprisingly competitive with paid alternatives for RAG use cases

---

## Week 1 Output
[View full notebook output (PDF)](Results/Week1_Output.pdf)

---

## 📬 About

Built by me as a self-learning project to break into AI/ML engineering.
Follow the build on GitHub — new notebooks added weekly!
