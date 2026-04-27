from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_document import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# load documents
documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
documents = load_documents(folder_path=documents_dir)

# create document chunker
def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

all_chunks = create_chunks(documents=documents)


# create vector store function
def create_vector_store(chunks):
    # create embedding model
    document_embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    # create vector store
    vector_store = FAISS.from_documents(documents=chunks, embedding=document_embedding_model)
    # store embeddings in a local FAISS 
    DB_FAISS_PATH="vectorstore/db_faiss"
    vector_store.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved to {DB_FAISS_PATH}")
    return vector_store