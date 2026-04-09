from backend.services.rag import get_retriever

def test_queries():
    retriever = get_retriever()
    
    queries = [
        "how many anexsure is tehre",  # Typo
        "annexures how many are there" # Correct
    ]
    
    print(f"Retriever Config: weights={retriever.weights}")
    # Verify underlying retriever K (we can't access easily without inspecting object)
    
    for q in queries:
        print(f"\n--- Query: '{q}' ---")
        docs = retriever.invoke(q)
        print(f"Retrieved {len(docs)} docs:")
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', '?')
            page = doc.metadata.get('page', '?')
            snippet = doc.page_content.replace('\n', ' ')[:100]
            print(f"  {i+1}. [Page {page}] {snippet}...")

if __name__ == "__main__":
    test_queries()
