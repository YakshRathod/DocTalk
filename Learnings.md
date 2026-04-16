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
