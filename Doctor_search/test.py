from service.rag.load_document import load_doctor_documents
from service.rag.langchain_document import to_langchain_documents
from service.rag.vector_store import create_faiss_vectorstore
from service.rag.search.hybrid import hybrid_langchain_retriever

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# todo: rag things here
# load the doctors document from the local system
# documents = load_doctor_documents("documents/doctors.json")

# # convert the raw documents to langchain documents
# langchain_documents = to_langchain_documents(documents)

# print("langchain_documents: ", langchain_documents)

# # create the vectorstore
# vectorstore = create_faiss_vectorstore(langchain_documents)

# # print the vectorstore
# print("vectorstore: ", vectorstore)

# load the vectorstore
vectorstore = FAISS.load_local("vectorstore/db_faiss", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

# test the vectorstore
# similar_docs = vectorstore.similarity_search("What is the name of the doctor who specializes in Cardiology?", k=2)
# print("similar_docs: ", similar_docs)


# load the doctors document from the local system
documents = load_doctor_documents("documents/doctors.json")

    # convert the raw documents to langchain documents
langchain_documents = to_langchain_documents(documents)
print("langchain_documents: ", langchain_documents)


# test hybrid retriever
result = hybrid_langchain_retriever(
    query="Medicine",
    lc_documents=langchain_documents,
    vectorstore=vectorstore
)

print("Strategy:", result["strategy"])
for d in result["documents"]:
    print(d.metadata)
