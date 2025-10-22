from fastapi import FastAPI, Query, Form, File, UploadFile
from pydantic import BaseModel
from decouple import config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
import glob
import shutil

app = FastAPI()

# Configuration
SECRET_KEY = config('OPENAI_API_KEY')
DATA_FOLDER = "./ihelp_data"
CHROMA_FOLDER = "./ihelp_chroma_db"

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)

def delete_files():
    """Deletes all files inside the data folder"""
    try:
        for file in glob.glob(os.path.join(DATA_FOLDER, "*")):
            os.remove(file)
        return {'success': True}
    except Exception as e:
        return {
            'code': 401,
            'success': False,
            'message': f'Delete files error: {str(e)}',
        }
        
def delete_chroma():
    """Deletes and recreates the chroma database folder"""
    try:
        if os.path.exists(CHROMA_FOLDER):
            shutil.rmtree(CHROMA_FOLDER)  
        os.makedirs(CHROMA_FOLDER, exist_ok=True)
        return {'success': True}
    except Exception as e:
        return {
            'code': 401,
            'success': False,
            'message': f'Error deleting chroma folder: {str(e)}',
        }

def preprocess_query(query: str) -> str:
    """Enhance query for better retrieval"""
    # Remove extra whitespace
    query = ' '.join(query.split())
    # You can add more preprocessing like expanding acronyms, etc.
    return query

def create_optimized_text_splitter():
    """Create text splitter with better parameters for FAQ documents"""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better precision
        chunk_overlap=200,  # More overlap to preserve context
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        add_start_index=True  # Track original position
    )

@app.post("/api/v1/faq/upload")
async def api(file: UploadFile):
    try:
        # Delete existing data
        delete_files()
        delete_chroma()

        # Save uploaded file
        file_path = os.path.join(DATA_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load document
        loader = PyPDFLoader(file_path)
        tech_doc = loader.load()

        # Better text splitting
        text_splitter = create_optimized_text_splitter()
        tech_split_doc = text_splitter.split_documents(tech_doc)

        # Create embeddings
        embeddings_model = OpenAIEmbeddings(
            openai_api_key=SECRET_KEY,
            model="text-embedding-3-small"  # Faster and cheaper
        )

        # Create vector database with correct path
        db = Chroma.from_documents(
            documents=tech_split_doc,
            embedding=embeddings_model,
            persist_directory=CHROMA_FOLDER,  # Use constant instead of hardcoded path
            collection_name="faq_collection"
        )

        return {
            'code': 200,
            'success': True,
            'message': 'Successfully uploaded file',
            'chunks_created': len(tech_split_doc)
        }

    except Exception as e:
        return {
            'code': 401,
            'success': False,
            'message': str(e),
        }

@app.post("/api/v1/faq/search")
async def search(query: str = Form()):
    try:
        # Check if database exists
        if not os.path.exists(CHROMA_FOLDER) or not os.listdir(CHROMA_FOLDER):
            return {
                'code': 400,
                'success': False,
                'message': 'No documents uploaded yet. Please upload a document first.',
            }

        # Preprocess query
        processed_query = preprocess_query(query)

        # Create embeddings model
        embeddings_model = OpenAIEmbeddings(
            openai_api_key=SECRET_KEY,
            model="text-embedding-3-small"
        )

        # Load vector store
        db = Chroma(
            persist_directory=CHROMA_FOLDER,  # Use constant
            embedding_function=embeddings_model,
            collection_name="faq_collection"
        )

        # Multi-query retrieval strategy
        # First: Get more candidates with MMR
        base_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # Get more initial results
                "fetch_k": 30,  # Fetch even more for diversity
                "lambda_mult": 0.7  # Favor relevance slightly more
            }
        )

        # Second: Use contextual compression for reranking
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cheaper for compression
            temperature=0,
            openai_api_key=SECRET_KEY
        )
        
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # Retrieve relevant chunks
        relevant_chunks = compression_retriever.invoke(processed_query)

        # Also try similarity search as fallback
        if len(relevant_chunks) == 0:
            relevant_chunks = db.similarity_search(
                processed_query,
                k=5
            )

        # Process results
        if len(relevant_chunks) > 0:
            # Combine all relevant context
            context = "\n\n---\n\n".join([
                f"[Chunk {i+1}]\n{chunk.page_content}" 
                for i, chunk in enumerate(relevant_chunks)
            ])

            # Improved prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert FAQ assistant. Your role is to provide accurate, 
                helpful answers based ONLY on the provided context from the documentation.
                
                Rules:
                1. Answer based ONLY on the provided context
                2. If the context contains relevant information, provide a clear, concise answer
                3. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the uploaded documents."
                4. Be specific and cite information from the context when possible
                5. If multiple relevant points exist, organize them clearly
                6. Do not make up or infer information not in the context"""),
                
                ("user", """Context from documentation:
                {context}
                
                User Question: {query}
                
                Please provide a helpful answer based on the context above.""")
            ])

            # Generate final prompt
            final_prompt = prompt_template.format_messages(
                context=context,
                query=query
            )

            # Generate response with better model
            response_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,  # Slight creativity while staying factual
                openai_api_key=SECRET_KEY
            )
            response = response_llm.invoke(final_prompt)

            return {
                'code': 200,
                'success': True,
                'message': 'Successfully retrieved answer',
                'data': {
                    "content": response.content,
                    "chunks_used": len(relevant_chunks),
                    "sources": [
                        {
                            "content": chunk.page_content[:200] + "...",
                            "page": chunk.metadata.get('page', 'N/A')
                        }
                        for chunk in relevant_chunks[:3]  # Show top 3 sources
                    ]
                }
            }
        else:
            return {
                'code': 200,
                'success': True,
                'message': 'No relevant information found',
                'data': {
                    "content": "I couldn't find any relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload a document that contains this information.",
                    "chunks_used": 0
                }
            }

    except Exception as e:
        return {
            'code': 401,
            'success': False,
            'message': str(e),
        }

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    db_exists = os.path.exists(CHROMA_FOLDER) and bool(os.listdir(CHROMA_FOLDER))
    return {
        "status": "healthy",
        "database_initialized": db_exists
    }