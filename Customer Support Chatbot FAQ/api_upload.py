from load_document import load_documents
from build_vector_store import create_chunks, create_vector_store
from fastapi import FastAPI, File, UploadFile
from typing import List, Annotated
import os


app = FastAPI(
    title="iHelpBD AI Chatbot API",
    description="API for uploading documents (pdf, txt, docx) and creating vector store for RAG-based question-answering.",
    version="1.0"
)

# API endpoint to handle document (files: pdf, txt, docx) uploading
@app.post("/upload")
def upload_documents(files: Annotated[List[UploadFile], File(...)]):
    documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    os.makedirs(documents_dir, exist_ok=True)
    
    # if no files uploaded, return error message
    if not files:
        return {
            "error": "No files uploaded. Please upload at least one document (pdf, txt, docx)."
        }
    
    uploaded_filenames = []
    
    # save uploaded files in the document directory
    for file in files:
        folder_path = os.path.join(documents_dir, file.filename)
        # if file name ends with .pdf, .txt, or .docx, save the file. Otherwise, skip and return unsupported file type message
        if file.filename.endswith('.pdf') or file.filename.endswith('.txt') or file.filename.endswith('.docx'):
            with open(folder_path, "wb") as f:
                f.write(file.file.read())
        else:
            return {
                "error": f"Unsupported file type for {file.filename}. Please upload a PDF, TXT, or DOCX file."
            }

        uploaded_filenames.append(file.filename)
    
    # load the uploaded documents
    documents = load_documents(folder_path=documents_dir)
    
    
    # create chunks of documents
    document_chunks = create_chunks(chunk_size=600, chunk_overlap=150, documents=documents)
    
    # create vector store
    vector_store = create_vector_store(chunks=document_chunks)
    
    return {
        "message": f"{len(uploaded_filenames)} document(s) uploaded successfully.",
        "uploaded_files": uploaded_filenames,
        "Length of loaded document pages": len(documents)
    }