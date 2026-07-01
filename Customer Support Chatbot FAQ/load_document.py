import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Create documents directory if it doesn't exist
documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
os.makedirs(documents_dir, exist_ok=True)

# document loader function
def load_documents(folder_path:str):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

# test the document loader function
# documents = load_documents(folder_path=documents_dir)
# print("Loaded documents: ", documents)
# print("Length of loaded document pages: ", len(documents))