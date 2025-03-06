import shutil
from grpc import Status
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
from fastapi import FastAPI , File, HTTPException, UploadFile, Form
from pydantic import BaseModel

app = FastAPI()

# output response model
class ProcessingResponse(BaseModel):
    status: str
    message: str
    details: dict


# Function for upload file from user
@app.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile]):
    try:
        # Create documents directory if it doesn't exist
        documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
        os.makedirs(documents_dir, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            # check if the file is a pdf file
            if not file.filename.endswith(('.pdf')):
                    uploaded_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": "Unsupported file type. Only PDF allowed."
                    })
                    continue
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


        # Check if any file was uploaded successfully
        if not any(f["status"] == "success" for f in uploaded_files):
                raise HTTPException(
                    status_code=Status.HTTP_400_BAD_REQUEST,
                    detail="No files were uploaded successfully"
                )
        
        # Load documents
        documents = load_documents(documents_dir)
        if not documents:
                raise HTTPException(
                    status_code=Status.HTTP_404_NOT_FOUND,
                    detail="No valid documents found after upload"
                )
        

                # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_documents = text_splitter.split_documents(documents)

        # Initialize embedding model
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )


        current_dir = os.path.dirname(os.path.abspath(__file__))
        persistent_directory = os.path.join(current_dir, "db", "chroma_db")
        # Create and persist vector store
        if os.path.exists(persistent_directory):
            shutil.rmtree(persistent_directory)  # Remove existing database
        
        vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=embedding_model,
            persist_directory=persistent_directory
        )
        
        return ProcessingResponse(
            status="success",
            message="Files uploaded and processed successfully",
            details={
                "uploaded_files": uploaded_files,
                "documents_processed": len(documents),
                "chunks_created": len(split_documents),
                "database_location": persistent_directory
            }
        )
        
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during processing: {str(e)}"
        )
    
    

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


