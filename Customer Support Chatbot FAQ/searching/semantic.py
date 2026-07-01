def semantic_retriever(vectorstore, query, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    results = docs

    return results
