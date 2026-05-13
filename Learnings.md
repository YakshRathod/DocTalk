# Learnings & Observations

A running log of things I learned, problems I ran into, and decisions I made while building DocTalk.

---

## Week 1 — Foundations

### Chunk Size Matters
Dense fact chunks containing multiple unrelated facts (temperature, moons, rings) don't 
retrieve well for specific queries. A chunk with 5 different facts will score mediocrely 
for all 5 queries rather than perfectly for any one of them. Smaller chunk sizes or 
fact-level chunking would help with precision.

### Reference Sections Pollute the Index
When testing with the Solar System Wikipedia PDF, reference chunks (containing DOI links, 
S2CID codes, JSTOR links) were consistently outranking actual content chunks. This is 
because reference chunks are dense with numbers and abbreviations that share semantic 
overlap with content queries. The fix is upstream of the index — either remove reference 
chunks before indexing or use a reranker at query time.

Switched to the NASA Mars fact sheet PDF which has no references section — retrieval 
improved dramatically.

### HuggingFace vs OpenAI Embeddings
Chose HuggingFace all-MiniLM-L6-v2 over OpenAI embeddings. It is a BERT-based model 
fine-tuned for semantic similarity, completely free, runs locally, and works offline. 
Quality is competitive with paid alternatives for RAG use cases. No API key needed.

### Why LangChain
Used LangChain as glue code to connect PDF loading, chunking, embeddings, and FAISS 
without writing boilerplate. Tradeoff is it abstracts away what's happening underneath. 
Plan to rewrite the core pipeline without LangChain in Week 5 to cement understanding.

### Relevance Filtering
Explored using similarity scores to filter out low quality retrievals instead of blindly 
returning top k chunks. FAISS uses L2 distance — lower score means more similar, which 
is counterintuitive. Found a natural score gap between genuinely relevant chunks and 
weakly related ones.

Decided against hardcoded pattern filtering (DOI, JSTOR, S2CID) because it only works 
for academic documents, not a general purpose app. Better solutions in order of 
sophistication:
1. Score threshold at query time (explored in Week 1, not implemented)
2. Contextual chunk enrichment using LLM (explored, not implemented — scaling concerns)
3. Cross-encoder reranker (planned for Week 5)

### Contextual Enrichment Tradeoff
Prepending LLM-generated descriptions to chunks before embedding improves retrieval 
quality significantly. However for a general purpose app with arbitrary documents, this 
means one LLM call per chunk at index time — potentially thousands of calls for large 
documents. Not practical for a general purpose app. Better suited for fixed curated 
knowledge bases. Reranking is the more scalable solution.

### Multi-Document Indexing
Current setup uses a single FAISS index which gets overwritten on each new upload. 
Plan to give each document its own index folder using the filename as the folder name. 
Two approaches for the final Streamlit app:
- Option 1: One index per document, user selects which to query (Week 4)
- Option 2: Merge all documents into one index, query across all (Week 5 upgrade)

---

## Week 2 — LLM Integration

### Groq Model Deprecation
llama3-8b-8192 was decommissioned by Groq mid-project. Switched to 
llama-3.3-70b-versatile. Lesson: always check provider deprecation notices before 
starting a project and pin model versions where possible.

### LangChain Import Instability
Multiple LangChain modules had moved or been removed in the current version — 
langchain.prompts, langchain.schema, langchain.retrievers all threw ModuleNotFoundError. 
Correct imports are now in langchain_core. LangChain's rapid iteration makes it 
unreliable for long-term projects without pinning versions in requirements.txt.

### ContextualCompressionRetriever Removed
LangChain completely removed ContextualCompressionRetriever and LLMChainExtractor from 
the current version. Implemented custom compression function instead — extracts verbatim 
relevant text from each chunk using the LLM before passing to the answer prompt.

### Compression Introduced Hallucination
Custom compression caused the LLM to generate summaries instead of extracting verbatim 
text, even when explicitly instructed not to. For example, querying temperature returned 
a fabricated "The cold temperatures..." instead of the actual "-87 to -5 deg C" from 
the document. This is a known risk with LLM-based compression on short factual text.

### Chunk Size Was the Real Problem
Reducing chunk_size from 500 to 200 was the single most impactful change in the entire 
project. Dense fact chunks were the root cause of poor retrieval — smaller chunks give 
each fact its own vector, dramatically improving precision. The temperature chunk went 
from not appearing in top 10 results to appearing at rank 2 after the size reduction.

### Simpler Pipeline Outperformed Complex One
Contextual compression was abandoned after it introduced hallucination and added 
unnecessary complexity. The simpler pipeline — retrieve then answer directly — 
outperformed the compression pipeline on every query. Complexity is not always better. 
Fix the data quality first before adding layers on top.

### Prompt Engineering Observations
Temperature=0 on the LLM produces consistent, factual answers with no creativity. 
The "answer strictly from context" instruction correctly caused the model to say 
"I could not find an answer" for the Viking 1 exact date query — the document only 
mentions 1976 without a specific date. This is the correct honest behaviour for a 
grounded Q&A system.

### Understanding LangChain Abstractions
Explored what LangChain's Runnables abstract away:
- RunnablePassthrough — passes input unchanged, equivalent to just using the variable directly
- RunnableLambda — wraps a plain Python function so it can be used as a chain step
- Pipe operator | — chains steps together, output of one becomes input of the next
- StrOutputParser — extracts .content from the LLM's ChatMessage response object

The full RAG chain without LangChain is 5 lines of plain Python. LangChain's declarative 
syntax is cleaner to read but hides what's happening. Plan to rewrite without LangChain 
in Week 5.

---

## Week 3 — Refinements

### pdfplumber Only Reads Real Tables
Table extraction works only if the source document contains an actual structured table — rows, columns, cells. If content is visually formatted to look like a table using two-column text layout (common in NASA fact sheets and marketing PDFs), pdfplumber won't detect it. The underlying text is still extracted correctly by PyMuPDF, so no data is lost — but the table-specific extraction adds no value for these documents.

### RAG Struggles with Comparative Table Queries
RAG handles direct fact lookup from tables well — "what is the orbital velocity of Neptune" retrieves the right row and answers correctly. But comparative reasoning across multiple rows fails — "which planet has the longest orbital period" requires the LLM to see all 8 rows simultaneously, which chunking and top-k retrieval prevent. Keeping tables as single unchunked documents helps, but the retriever still only returns top k results so the full table may not always be included.

### Per-Document Indexing Prevents Overwrites
Moving from a single shared FAISS index to per-document indexes means new uploads never overwrite existing ones. Each document gets its own folder. FAISS's built-in `merge_from` method combines selected indexes at query time in a single line — making multi-document querying trivial without rebuilding anything.

### Table Chunks Must Not Be Split
When table data was passed through the standard `RecursiveCharacterTextSplitter`, rows were cut mid-way — a row containing temperature, moons, and rings would get split across two chunks, losing the relationship between the header and the value. This caused the LLM to return "I could not find an answer" even when the data was technically in the index.

Fixed by separating text documents and table documents before chunking — text docs go through the splitter normally, table docs are added to the index as-is without splitting. Each table is one chunk, keeping all rows and their headers together.

---

## Week 4 — Table Extraction and Streamlit UI

### Streamlit UI Logic Must Be Tied to Explicit Triggers
Any logic that runs freely in the script executes on every rerun — every button click, 
dropdown change, or interaction. This caused duplicate warnings and indexing logic to 
fire repeatedly without user intent. Fixed by gating all stateful operations behind 
explicit trigger buttons so nothing runs unless the user deliberately initiates it.

### Cached Resources Persist After Deletion
Deleting a FAISS index folder from disk didn't remove it from the app because Streamlit's 
resource cache still held a reference to it. The deleted index kept appearing in the 
available documents list. Fixed by calling `st.cache_resource.clear()` before rerunning 
after deletion — forces Streamlit to drop all cached objects and reload from disk.

### Streamlit Form Hints Can Be Hidden with CSS
The "Press Enter to submit form" hint that appears inside Streamlit forms cannot be 
removed through the API — it's hardcoded into the component. Hidden it using CSS by 
targeting the `[data-testid="InputInstructions"]` element and setting `display: none`. 
A small thing but important for a clean demo presentation.

### File Uploader Needs a Dynamic Key to Reset
After indexing, the file uploader retained the uploaded files across reruns — the files 
stayed visible in the sidebar even after processing was complete. Streamlit doesn't 
provide a direct reset method for the uploader. Fixed by assigning a dynamic key to the 
uploader using a session state counter — `key=f"uploader_{st.session_state['uploader_key']}"` 
— and incrementing the counter after each successful index. Streamlit treats a widget 
with a new key as a brand new widget with fresh state, effectively clearing the uploaded 
files.

### Sidebar Indentation Is Critical in Streamlit
All sidebar elements must be properly indented inside the `with st.sidebar:` block. A 
single indentation error caused the file uploader to render in the main content area 
instead of the sidebar, breaking the entire layout. Python's strict indentation rules 
apply to Streamlit layout blocks just as they do to any other code.

## Week 5 — Folder Indexing and Reranker

### Folder Indexing Uses Merge Not Rebuild
When adding a new document to a folder, merging the new document's index into the existing folder index using FAISS merge_from is faster than rebuilding the entire folder from scratch. The tradeoff is that removing a document from a folder requires a full rebuild since FAISS doesn't support vector deletion. This is acceptable because additions are frequent and deletions are rare.

### Cross-Encoder Reranker Fixes Retrieval Failures
Adding a cross-encoder reranker as a second stage after FAISS retrieval improved results on 2 out of 8 test queries without hurting any others. The reranker never made results worse — when FAISS was correct, the reranker agreed. When FAISS returned irrelevant chunks (moons of Mars, Neptune velocity), the reranker correctly promoted the relevant ones.

### k=50 Required for Comparative Table Queries
With k=20, the full orbital periods table chunk was not appearing in the reranker's candidate pool for queries like "which planet has the longest orbital period". Increasing to k=50 brought the table chunk to rank 2, giving the LLM enough context to reason across all planets and answer correctly. The cost is negligible — FAISS search over 50 candidates is still milliseconds.

### RAG Has a Hard Limit on Comparative Reasoning
Even with the reranker and k=50, comparative reasoning only works when the full data fits in a single retrievable chunk. If the answer requires synthesising information across many separate chunks, RAG will still struggle. The right solution for these cases is map-reduce — ask the LLM to answer from each section independently then combine — but that's expensive and out of scope for this project.
