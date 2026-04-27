from service.rag.load_document import load_doctor_documents
from service.rag.langchain_document import to_langchain_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# todo: rag things here
    # load the doctors document from the local system
documents = load_doctor_documents("documents/doctors.json")

    # convert the raw documents to langchain documents
langchain_documents = to_langchain_documents(documents)
    # print("langchain_documents: ", langchain_documents)
    
    
    # load the vectorstore
vectorstore = FAISS.load_local(
        "vectorstore/db_faiss", 
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
        )