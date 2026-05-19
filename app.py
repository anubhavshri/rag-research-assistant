import streamlit as st
import tempfile
import fitz
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

st.title("📄 Research Paper Analyzer")

# --- Upload PDFs ---
uploaded_files = st.file_uploader(
    "Upload max 3 PDFs", type=["pdf"], accept_multiple_files=True
)

paper_dict = {}

if uploaded_files:
    uploaded_files = uploaded_files[:3]
    
    for i, file in enumerate(uploaded_files):
        pid = f"P{i+1}"
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        paper_dict[pid] = temp.name
    
    st.success(f"{len(paper_dict)} paper(s): {list(paper_dict.keys())}")

# --- Extract + CLEAN text ---
documents = []

if paper_dict:
    for pid, path in paper_dict.items():
        doc = fitz.open(path)
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # CLEAN TEXT
            lines = text.split("\n")
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                
                if len(line) < 20:
                    continue
                if "et al" in line.lower():
                    continue
                if "references" in line.lower():
                    continue
                if line.startswith("["):
                    continue
                
                clean_lines.append(line)
            
            text = " ".join(clean_lines)
            
            documents.append({
                "paper_id": pid,
                "page": page_num+1,
                "text": text
            })

# --- Chunking ---
chunks = []

for d in documents:
    text = d["text"]
    
    parts = [text[i:i+500] for i in range(0, len(text), 500)]
    
    for p in parts:
        chunks.append({
            "paper_id": d["paper_id"],
            "page": d["page"],
            "text": p
        })

# --- Setup RAG ---
@st.cache_resource
def setup_rag(chunks):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)
    
    client = chromadb.Client()

    # ✅ FIX: delete old collection
    try:
        client.delete_collection("rag")
    except:
        pass

    collection = client.create_collection("rag")
    
    for i, c in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            embeddings=[embeddings[i]],
            documents=[c["text"]],
            metadatas=[{
                "paper_id": c["paper_id"],
                "page": c["page"]
            }]
        )
    
    bm25 = BM25Okapi([t.split() for t in texts])
    
    return embed_model, collection, bm25

if chunks:
    embed_model, collection, bm25 = setup_rag(chunks)

# --- Hybrid Search ---
def hybrid_search(query, top_k=5):
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    query_embedding = embed_model.encode([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    vector_ids = results["ids"][0]
    
    combined = []
    for i in range(len(chunks)):
        score = bm25_scores[i]
        if str(i) in vector_ids:
            score += 1.0
        combined.append((i, score))
    
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
    return [chunks[i] for i, _ in combined[:top_k]]

# --- Llama Answer ---
def generate_answer(query, retrieved):
    
    context = "\n\n".join([c["text"] for c in retrieved])
    
    papers = list(set([c["paper_id"] for c in retrieved]))
    papers_str = ", ".join(papers)
    
    prompt = f"""
You are analyzing research papers.

Available papers: {papers_str}

STRICT RULES:
- Only use these paper IDs: {papers_str}
- Do NOT create extra papers
- Do NOT use author names

Context:
{context}

Question:
{query}

Answer per paper clearly.
"""
    
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    
    return res.json()["response"]

# --- UI ---
query = st.text_input("Ask question")

if st.button("Search Answer"):
    st.write(generate_answer(query, hybrid_search(query)))

if st.button("GAP"):
    st.write(generate_answer("Find research gaps for each paper", hybrid_search("limitations gap")))

if st.button("Methodology"):
    st.write(generate_answer("Explain methodology of each paper", hybrid_search("methodology model training dataset")))

if st.button("Results"):
    st.write(generate_answer("Extract results of each paper", hybrid_search("results accuracy precision recall")))

if st.button("Hindi Summary"):
    st.write(generate_answer("Summarize each paper and translate to Hindi", hybrid_search("abstract summary")))