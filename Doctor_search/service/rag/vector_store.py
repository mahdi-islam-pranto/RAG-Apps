from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_faiss_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    DB_FAISS_PATH="vectorstore/db_faiss"
    print("vectorstore created: ", vectorstore)
    # save the vectorstore
    vectorstore.save_local(DB_FAISS_PATH)
    return vectorstore


# load the vectorstore
