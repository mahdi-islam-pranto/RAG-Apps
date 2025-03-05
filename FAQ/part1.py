from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
import os

## fast api
from fastapi import FastAPI , File, UploadFile, Form
from pydantic import BaseModel

app = FastAPI()


# Function for upload file from user

@app.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile]):
    # Create documents directory if it doesn't exist
    documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    os.makedirs(documents_dir, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        try:
            file_path = os.path.join(documents_dir, file.filename)
            with open(file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)
            uploaded_files.append({
                "filename": file.filename,
                "status": "success"
            })
        except Exception as e:
            uploaded_files.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"uploaded_files": uploaded_files}
    

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents


# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Added more separators
)


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# The path to the text document to be processed
folder_path = os.path.join(current_dir, "documents")
# Load the documents from the folder
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")


# Split the documents into chunks
split_documents = text_splitter.split_documents(documents)
print(f"Split the documents into {len(split_documents)} chunks.")

# Print the first chunk
print(documents[0])


persistent_directory = os.path.join(current_dir, "db", "chroma_db")
# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Initialize the vector store
    # create Embeddings & store them in vector database
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        
    )

    document_embeddings = embedding_model.embed_documents([split.page_content for split in split_documents])
    print(f"Created embeddings for {len(document_embeddings)} document chunks.")
    print("\n--- Finished creating embeddings ---")


    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    collection_name = "my_collection"
    vectorstore = Chroma.from_documents(
        
        split_documents,
        # embeddings,
        embedding_model,
        persist_directory=persistent_directory
)
    print("Vector store created and persisted to './chroma_db'")

    print("\n--- Finished creating vector store ---")


else:
    print("Persistent directory exists.")






    



