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
