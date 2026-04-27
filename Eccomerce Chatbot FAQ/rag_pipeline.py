
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from searching.hybrid import hybrid_langchain_retriever


# load vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
document_embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

loaded_vector_store = FAISS.load_local(
    folder_path=DB_FAISS_PATH,
    embeddings=document_embedding_model,
    allow_dangerous_deserialization=True
    )

# create retriever and test
# similar_docs = loaded_vector_store.similarity_search("names of the web developers of iHelpBD")
# print(similar_docs)

# hybrid search retriever
# results = hybrid_langchain_retriever(lc_documents=all_chunks,vectorstore=loaded_vector_store,query="Tell me all developers name of iHelpBD")
# print(results)

