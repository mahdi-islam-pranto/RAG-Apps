from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
from langchain.embeddings.base import Embeddings
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()

# Load the document loader

# Step 1: Load raw PDF(s)
# Create documents directory if it doesn't exist
documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
os.makedirs(documents_dir, exist_ok=True)

# load documents function

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

documents = load_documents(folder_path=documents_dir)
print("Length of PDF pages: ", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))
print(text_chunks[2])



# Load BanglaBERT tokenizer and model
model_name = "sagorsarker/bangla-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# Step 3: Create Vector Embeddings 

class BanglaBERTEmbeddings(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
        return embeddings.numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Initialize the custom BanglaBERT embedding class
bangla_bert_embeddings = BanglaBERTEmbeddings(model, tokenizer)


# Step 4: Create Vector Store


# Generate embeddings and create a FAISS vector store
DB_FAISS_PATH="vectorstore/db_faiss"
vector_store = FAISS.from_documents(text_chunks, embedding=bangla_bert_embeddings)


vector_store.save_local(DB_FAISS_PATH)
