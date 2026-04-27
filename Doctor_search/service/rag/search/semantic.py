

def semantic_retriever(vectorstore, query, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    results = [f"{doc.metadata['name']} {doc.metadata['department']}" for doc in docs]

    return results
