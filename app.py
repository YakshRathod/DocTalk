import streamlit as st
import os
import shutil
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocTalk",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0f0f0f;
    color: #e8e8e8;
}

section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #222;
}

.block-container {
    padding: 2rem 2.5rem;
    max-width: 900px;
}

h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #f0f0f0;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

h2, h3 {
    font-family: 'DM Mono', monospace;
    font-weight: 400;
    color: #d0d0d0;
}

.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.answer-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #6ee7b7;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e0e0e0;
    margin: 1rem 0;
}

.source-item {
    background: #161616;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    line-height: 1.5;
}

.source-label {
    color: #6ee7b7;
    font-weight: 500;
}

.tag {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 2px;
    padding: 0.1rem 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #666;
    margin-right: 0.3rem;
}

.tag.table {
    border-color: #6ee7b7;
    color: #6ee7b7;
}

.doc-item {
    background: #161616;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 0.5rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #888;
    margin: 0.3rem 0;
}

.stButton > button {
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    padding: 0.4rem 1rem;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: #222;
    border-color: #6ee7b7;
    color: #6ee7b7;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    color: #e0e0e0;
    font-family: 'DM Sans', sans-serif;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
}

.stRadio > div {
    gap: 0.5rem;
}

.stSuccess {
    background: #0d1f18;
    border: 1px solid #6ee7b7;
    color: #6ee7b7;
}

.stWarning {
    background: #1f1a0d;
    border: 1px solid #fbbf24;
}

.stError {
    background: #1f0d0d;
    border: 1px solid #f87171;
}

hr {
    border-color: #222;
    margin: 1.5rem 0;
}

.stSpinner > div {
    border-top-color: #6ee7b7 !important;
}

.stForm small {
    display: none !important;
}

[data-testid="InputInstructions"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DOCS_DIR = "uploaded_docs"
STANDALONE_DIR = "faiss_index/standalone"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(STANDALONE_DIR, exist_ok=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "selected_indexes" not in st.session_state:
    st.session_state.selected_indexes = []

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

# ── Core functions ─────────────────────────────────────────────────────────────
def extract_tables(filepath):
    tables_text = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    rows = []
                    for row in table:
                        cleaned = [cell.strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_text = "\n".join(rows)
                    if table_text.strip():
                        tables_text.append({"page": page_num, "content": table_text})
    except Exception:
        pass
    return tables_text

def tables_to_documents(tables, source_path):
    docs = []
    for table in tables:
        doc = Document(
            page_content=table["content"],
            metadata={"source": source_path, "page": table["page"], "type": "table"}
        )
        docs.append(doc)
    return docs

def load_document(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(filepath)
        docs = loader.load()
        tables = extract_tables(filepath)
        if tables:
            table_docs = tables_to_documents(tables, filepath)
            docs.extend(table_docs)
    elif ext == ".docx":
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
    elif ext == ".txt":
        loader = TextLoader(filepath)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not docs:
        raise ValueError("Document is empty or could not be parsed.")
    total_text = "".join([d.page_content for d in docs]).strip()
    if len(total_text) < 100:
        raise ValueError("Document contains too little text to be useful.")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    text_docs = [d for d in docs if d.metadata.get("type") != "table"]
    table_docs = [d for d in docs if d.metadata.get("type") == "table"]
    chunks = splitter.split_documents(text_docs)
    chunks.extend(table_docs)
    return chunks

def build_index(chunks, index_path, embeddings):
    os.makedirs(index_path, exist_ok=True)
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [c.page_content for c in batch]
        all_embeddings.extend(embeddings.embed_documents(batch_texts))
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vs = FAISS.from_embeddings(list(zip(texts, all_embeddings)), embeddings, metadatas=metadatas)
    vs.save_local(index_path)
    return vs

def list_indexes():
    if not os.path.exists(STANDALONE_DIR):
        return []
    return [
        name for name in os.listdir(STANDALONE_DIR)
        if os.path.isdir(os.path.join(STANDALONE_DIR, name))
    ]

def load_vectorstore(selected, embeddings):
    if not selected:
        return None
    vs = FAISS.load_local(
        os.path.join(STANDALONE_DIR, selected[0]),
        embeddings,
        allow_dangerous_deserialization=True
    )
    for name in selected[1:]:
        other = FAISS.load_local(
            os.path.join(STANDALONE_DIR, name),
            embeddings,
            allow_dangerous_deserialization=True
        )
        vs.merge_from(other)
    return vs

def build_rag_chain(vectorstore, llm):
    template = """
You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not found in the context, say "I could not find an answer in the document."
Do not use any knowledge outside of the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    def format_chunks(docs):
        return "\n\n".join([
            f'[Page {doc.metadata.get("page", 0) + 1}] [{"table" if doc.metadata.get("type") == "table" else "text"}]\n{doc.page_content}'
            for doc in docs
        ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = (
        {"context": RunnableLambda(lambda q: retriever.invoke(q)) | format_chunks, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DocTalk")
    st.markdown('<p class="subtitle">document intelligence</p>', unsafe_allow_html=True)
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")

    # Upload
    st.markdown("**Upload Documents**")
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_files and groq_key:
        if st.button("Index Documents", key="index_btn"):
            st.session_state["pending_files"] = uploaded_files
            st.session_state["confirm_reindex"] = []
            for uploaded_file in uploaded_files:
                doc_name = os.path.splitext(uploaded_file.name)[0]
                index_path = os.path.join(STANDALONE_DIR, doc_name)
                if os.path.exists(index_path):
                    st.session_state["confirm_reindex"].append(doc_name)

        if "confirm_reindex" in st.session_state and st.session_state["confirm_reindex"]:
            for doc_name in st.session_state["confirm_reindex"]:
                st.warning(f"'{doc_name}' is already indexed. Reindex?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes", key=f"yes_{doc_name}"):
                        st.session_state["confirm_reindex"].remove(doc_name)
                        st.session_state[f"force_{doc_name}"] = True
                        st.rerun()
                with col2:
                    if st.button("No", key=f"no_{doc_name}"):
                        st.session_state["confirm_reindex"].remove(doc_name)
                        st.session_state[f"force_{doc_name}"] = False
                        st.rerun()

        if "pending_files" in st.session_state and not st.session_state.get("confirm_reindex"):
            embeddings = load_embeddings()
            for uploaded_file in st.session_state["pending_files"]:
                doc_name = os.path.splitext(uploaded_file.name)[0]
                index_path = os.path.join(STANDALONE_DIR, doc_name)

                if os.path.exists(index_path) and not st.session_state.get(f"force_{doc_name}", False):
                    continue

                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        filepath = os.path.join(DOCS_DIR, uploaded_file.name)
                        with open(filepath, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        docs = load_document(filepath)
                        chunks = chunk_documents(docs)
                        build_index(chunks, index_path, embeddings)
                        st.success(f"Indexed: {doc_name} ({len(chunks)} chunks)")
                    except ValueError as e:
                        st.error(str(e))

            del st.session_state["pending_files"]
            for uploaded_file in (uploaded_files or []):
                doc_name = os.path.splitext(uploaded_file.name)[0]
                st.session_state.pop(f"force_{doc_name}", None)
            st.session_state["uploader_key"] += 1
            st.rerun()
    st.markdown("---")

    # Available indexes
    indexes = list_indexes()
    if indexes:
        st.markdown("**Available Documents**")
        for idx in indexes:
            st.markdown(f'<div class="doc-item">📄 {idx}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Delete
        st.markdown("**Delete Index**")
        to_delete = st.selectbox("Select to delete", [""] + indexes, label_visibility="collapsed")
        if to_delete and st.button("Delete", key="delete_btn"):
            shutil.rmtree(os.path.join(STANDALONE_DIR, to_delete))
            st.success(f"Deleted: {to_delete}")
            st.cache_resource.clear()
            st.rerun()
# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<h1>DocTalk</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ask questions — get answers from your documents</p>', unsafe_allow_html=True)

indexes = list_indexes()

if not groq_key:
    st.info("Add your Groq API key in the sidebar to get started.")
elif not indexes:
    st.info("Upload a document in the sidebar to get started.")
else:
    embeddings = load_embeddings()
    llm = load_llm(groq_key)

    # Query mode
    mode = st.radio("Query mode", ["Single document", "Multiple documents"], horizontal=True)
    st.markdown("")

    if mode == "Single document":
        selected = st.selectbox("Select document", indexes)
        selected_indexes = [selected] if selected else []
    else:
        selected_indexes = st.multiselect("Select documents to query across", indexes)

    if selected_indexes:
        with st.spinner("Loading index..."):
            vs = load_vectorstore(selected_indexes, embeddings)
            chain, retriever = build_rag_chain(vs, llm)

        st.markdown("---")

        with st.form("query_form"):
            question = st.text_input("Ask a question", placeholder="What is...")
            ask = st.form_submit_button("Ask")

        if ask and question.strip():
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
                source_docs = retriever.invoke(question)

            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("**Sources**")
            for i, doc in enumerate(source_docs[:3]):
                source_file = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", 0) + 1
                doc_type = doc.metadata.get("type", "text")
                tag_class = "tag table" if doc_type == "table" else "tag"
                preview = doc.page_content[:150].replace("\n", " ")
                st.markdown(
                    f'<div class="source-item">'
                    f'<span class="source-label">[{i+1}]</span> '
                    f'<span class="{tag_class}">{doc_type}</span> '
                    f'{source_file} · page {page}<br>{preview}...'
                    f'</div>',
                    unsafe_allow_html=True
                )
        elif question == "":
            pass
        else:
            st.warning("Please enter a question.")
