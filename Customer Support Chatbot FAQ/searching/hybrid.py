from searching.fuzzy import fuzzy_retriever
from searching.semantic import semantic_retriever
# for test the retrievers
# from rag_pipeline import all_chunks, loaded_vector_store



def hybrid_langchain_retriever(query, lc_documents, vectorstore, k=5):
    all_results = []

    # Step 1: Fuzzy (best for names & typos)
    fuzzy_results = fuzzy_retriever(query, lc_documents, k)

    if fuzzy_results:
        all_results.extend(fuzzy_results)
        

    # Step 2: Semantic fallback
    semantic_results = semantic_retriever(vectorstore, query, k)

    if semantic_results:
        all_results.extend(semantic_results)
        
    return {
        "strategy": "hybrid",
        "documents": all_results
    }

# # test the hybrid retriever
# result = hybrid_langchain_retriever(
#     query="Tell me all developers name of iHelpBD",
#     lc_documents=all_chunks,
#     vectorstore=loaded_vector_store
# )

# print(f"retrieved documents: {result}")



    
