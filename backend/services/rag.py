import os
import re
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import math
from collections import Counter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional

class SimpleBM25Retriever(BaseRetriever):
    """
    A simple, dependency-free implementation of BM25.
    Replaces rank_bm25 to avoid environment issues.
    """
    vectorizer: Any = None
    docs: List[Document] = []
    k: int = 4
    corpus_size: int = 0
    avgdl: float = 0.0
    doc_freqs: List[Counter] = []
    idf: Dict[str, float] = {}
    doc_len: List[int] = []
    
    class Config:
        arbitrary_types_allowed = True
    
    
    def _tokenize(self, text: str) -> List[str]:
        # Split on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    def __init__(self, docs: List[Document], k: int = 4, **kwargs):
        # Initialize Pydantic Base
        super().__init__(docs=docs, k=k, **kwargs)
        
        self.corpus_size = len(docs)
        self.avgdl = 0
        if self.corpus_size > 0:
            self.avgdl = sum(len(self._tokenize(d.page_content)) for d in docs) / self.corpus_size
            
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        # Build index
        for doc in docs:
            tokens = self._tokenize(doc.page_content)
            self.doc_len.append(len(tokens))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for token in freqs:
                self.idf[token] = self.idf.get(token, 0) + 1
                
        # Calculate IDF
        for token, freq in self.idf.items():
            self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        
        k1 = 1.5
        b = 0.75
        
        for i in range(self.corpus_size):
            score = 0
            doc_len_norm = self.doc_len[i] / self.avgdl
            for token in query_tokens:
                if token not in self.doc_freqs[i]:
                    continue
                freq = self.doc_freqs[i][token]
                numerator = self.idf.get(token, 0) * freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len_norm)
                score += numerator / denominator
            scores[i] = score
            
        # Get top k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
        return [self.docs[i] for i in top_indices if scores[i] > 0] # Only return positive matches

# --- Replaced Imports ---
# from langchain_community.retrievers import BM25Retriever


from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Dict, Any
from collections import defaultdict

class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines results from multiple retrievers
    and re-ranks them using Weighted Reciprocal Rank Fusion (RRF).
    (Inlined to avoid langchain dependency issues)
    """
    retrievers: List[BaseRetriever]
    weights: List[float]

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        # Get results from all retrievers
        all_docs_list = []
        for retriever in self.retrievers:
            all_docs_list.append(retriever.invoke(query))
            
        # RRF Fusion
        rrf_score: Dict[str, float] = defaultdict(float)
        
        for doc_list, weight in zip(all_docs_list, self.weights):
            for rank, doc in enumerate(doc_list):
                # RRF score = weight / (rank + k)
                # k is usually 60
                score = weight / (rank + 60)
                rrf_score[doc.page_content] += score
        
        # Sort by score
        # Note: This is a simplified fusion that deduplicates by content
        # We need to map content back to documents
        content_to_doc = {doc.page_content: doc for docs in all_docs_list for doc in docs}
        
        sorted_contents = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)
        return [content_to_doc[content] for content in sorted_contents]



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db") 

_retriever = None


import shutil
import json

def _build_retriever(data_dir: str) -> Optional[object]:
    """
    Load all PDFs from data_dir, split into chunks, embed, and create a retriever.
    
    Implements Hybrid Search:
    - Vector Search (Chroma) for semantic understanding
    - BM25 for keyword/exact match (critical for case numbers like 15336-15337)
    
    Now checks for file changes and rebuilds index if needed.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 0. Check for file changes to decide if we need to rebuild
    if not os.path.exists(data_dir):
        print(f"[RAG] Data directory not found: {data_dir}")
        return None
        
    pdf_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.lower().endswith(".pdf")
    ])
    
    # Simple signature: filenames + sizes (or mtimes) AND configuration
    current_signature = {
        "files": {
            f: os.path.getmtime(os.path.join(data_dir, f)) 
            for f in pdf_files
        },
        "config": {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "chunk_size": 800,
            "chunk_overlap": 200
        }
    }
    
    signature_path = os.path.join(data_dir, "index_signature.json")
    rebuild_needed = False
    
    if os.path.exists(signature_path) and os.path.exists(CHROMA_DIR):
        try:
            with open(signature_path, "r") as f:
                saved_signature = json.load(f)
            
            if current_signature != saved_signature:
                print(f"[RAG] Index signature mismatch. \nSaved: {list(saved_signature.keys()) if isinstance(saved_signature, dict) else 'Unknown'}\nCurrent: {list(current_signature.keys())}\nRebuilding index...")
                rebuild_needed = True
            else:
                print("[RAG] Signature matches. Using existing index.")
        except Exception as e:
            print(f"[RAG] Error reading signature, forcing rebuild: {e}")
            rebuild_needed = True
    else:
         if os.path.exists(CHROMA_DIR):
             print("[RAG] Index exists but signature missing. Forcing rebuild to be safe.")
             rebuild_needed = True
    
    if rebuild_needed and os.path.exists(CHROMA_DIR):
        try:
            print(f"[RAG] Removing stale Chroma DB at {CHROMA_DIR}...")
            shutil.rmtree(CHROMA_DIR)
            print("[RAG] Old DB removed.")
        except Exception as e:
            print(f"[RAG] Failed to remove old DB: {e}. Attempting continue (might fail)...")

    # Check for existing Chroma DB (If we just deleted it, this will be False)
    vectordb = None
    if os.path.exists(CHROMA_DIR):
        print(f"[RAG] Loading existing Chroma DB from: {CHROMA_DIR}")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    
    # Save new signature if we are (re)building
    # We will do this after successful build

    
    # We need CHUNKS to build BM25 (it runs in-memory).
    # If we loaded Chroma from disk, we can extract documents from it.
    # If strictly starting fresh, we parse PDFs.
    
    docs_for_bm25 = []

    if vectordb:
        print("[RAG] Fetching documents from Vector DB to initialize BM25...")
        # Fetch all documents to build BM25 index
        # This is fast for a few thousand chunks
        exists_docs = vectordb.get()
        if exists_docs and exists_docs['documents']:
             # Reconstruct Document objects for BM25
             from langchain_core.documents import Document
             docs_for_bm25 = [
                 Document(page_content=txt, metadata=meta) 
                 for txt, meta in zip(exists_docs['documents'], exists_docs['metadatas'])
             ]
    
    if not docs_for_bm25:
        print(f"[RAG] No existing DB or empty. Scanning for PDFs in: {data_dir}")
        # 1. Provide a list of PDF files
        if not os.path.exists(data_dir):
            print(f"[RAG] Data directory not found: {data_dir}")
            return None
            
        pdf_files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.lower().endswith(".pdf")
        ]
        
        if not pdf_files:
            print("[RAG] No PDF files found in data folder.")
            return None

        print(f"[RAG] Found {len(pdf_files)} PDFs: {[os.path.basename(f) for f in pdf_files]}")
        
        all_chunks = []
        
        # 2. Iterate and chunk each PDF
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
    
        for pdf_path in pdf_files:
            try:
                print(f"[RAG] Processing: {os.path.basename(pdf_path)}")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                print(f"      - Loaded {len(docs)} pages.")
                chunks = splitter.split_documents(docs)
                # Add numeric tag to chunks containing numbers (4+ digits)
                for chunk in chunks:
                    if re.search(r"\d{4,}", chunk.page_content):
                        chunk.page_content = "NUM_REF: " + chunk.page_content
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"      - Error reading {pdf_path}: {e}")
    
        if not all_chunks:
            print("[RAG] No valid chunks extracted from PDFs.")
            return None

        print(f"[RAG] Total chunks from all files: {len(all_chunks)}")
        
        docs_for_bm25 = all_chunks
        
        # Build Vector DB if we didn't have it
        if not vectordb:
            print("[RAG] Building Chroma vector store and persisting to disk...")
            vectordb = Chroma.from_documents(
                all_chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
            )

    # --- Build Hybrid Retriever ---
    print(f"[RAG] Building BM25 Retriever from {len(docs_for_bm25)} chunks...")
    # bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    
    # Reduced k to 20 to fit within Groq Free Tier Rate Limits (6000 TPM)
    bm25_retriever = SimpleBM25Retriever(docs_for_bm25, k=20)

    bm25_retriever.k = 20  
    
    # Increase vector search depth too
    mask_retriever = vectordb.as_retriever(search_kwargs={"k": 20})
    
    print("[RAG] Initializing Ensemble Retriever (Hybrid Search)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mask_retriever],
        weights=[0.5, 0.5] # Equal weight to keywords and semantics
    )
    
    # Save the signature now that we're successful
    if rebuild_needed or not os.path.exists(signature_path):
        try:
            with open(signature_path, "w") as f:
                json.dump(current_signature, f)
            print("[RAG] Index signature saved.")
        except Exception as e:
            print(f"[RAG] Failed to save index signature: {e}")

    print("[RAG] Hybrid Retriever ready.")
    return ensemble_retriever


def get_retriever():
    """
    Singleton-style access to the retriever.
    Builds it on first use.
    """
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever(DATA_DIR)
    return _retriever
