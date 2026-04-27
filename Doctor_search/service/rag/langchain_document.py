from langchain_core.documents import Document

# convert the raw documents to langchain documents
def to_langchain_documents(raw_docs):
    lc_docs = []

    for doc in raw_docs:
        lc_docs.append(
            Document(
                page_content=doc["content"],
                metadata={
                    "doc_id": doc["doc_id"],
                    "name": doc["metadata"]["name"],
                    "department": doc["metadata"]["department"]
                }
            )
        )

    return lc_docs