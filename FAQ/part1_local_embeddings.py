from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
import os


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
    chunk_size=500,
    chunk_overlap=20,
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
    
    embedding_model = HuggingFaceEmbeddings(
        
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


    document_embeddings = embedding_model.embed_documents([split.page_content for split in split_documents])
    print(f"Created embeddings for {len(document_embeddings)} document chunks.")
    print(document_embeddings[0][:5])
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


