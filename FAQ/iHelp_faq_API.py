from fastapi import FastAPI, Query, Form, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

#necessary imports
from decouple import config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import os
import glob
import time
import shutil


# OpenAI API key
SECRET_KEY = config('OPENAI_API_KEY')
# Define the data folder
DATA_FOLDER = "./ihelp_data"
CHROMA_FOLDER = "./ihelp_chroma_db"

def delete_files():
    """Deletes all files inside the data folder using glob"""
    for file in glob.glob(os.path.join(DATA_FOLDER, "*")):
        try:
            os.remove(file)  # Remove each file
        except Exception as e:
            return {
                'code': 401,
                'success': False,
                'message': 'Deletes all files error',
            }
        
def delete_chroma():
    """Deletes all files inside the data folder and removes the folder itself."""
    try:
        if os.path.exists(CHROMA_FOLDER):
            shutil.rmtree(CHROMA_FOLDER)  
        os.makedirs(CHROMA_FOLDER, exist_ok=True) 
    except Exception as e:
        return {
            'code': 401,
            'success': False,
            'message': f'Error deleting files and folder: {str(e)}',
        }


@app.post("/api/v1/faq/upload")
async def api(file: UploadFile):
    try:

        # Delete all files before saving a new one
        delete_files()
        delete_chroma()

        # Save the uploaded file to the data folder
        #file_path= "./data/test.pdf"
        """ file_path = os.path.join(DATA_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read()) """
        
        file_path = os.path.join(DATA_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        # Document load
        loader = PyPDFLoader(file_path)
        tech_doc = loader.load()

        #text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # split the document
        tech_split_doc = text_splitter.split_documents(tech_doc)

        # embed data by OpenAI
        embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)

        #data based creation
        #get_chroma_db(tech_split_doc, embeddings_model)
        
        #db = Chroma.from_documents(tech_split_doc, embeddings_model, persist_directory='./chroma_db')
        ##db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
        db = Chroma.from_documents(tech_split_doc, embeddings_model, persist_directory='./chroma_db')
       #del db  # Free memory and close database connection


        return ({
            'code': 200,
            'success': True,
            'message':'successfully uploaded file',
        })

    except Exception as e:
        # Catch any errors and return a meaningful message
        return {
                'code': 401,
                'success': False,
                'message': str(e),
            }

# API for search 
@app.post("/api/v1/faq/search")
async def search(query: str = Form()):
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # The path to the database directory
        persistent_directory = os.path.join(current_dir, "chroma_db")

        # embedding model
        embeddings_model = OpenAIEmbeddings(openai_api_key=SECRET_KEY)

        # load the existing vector store with the embeddings
        db = Chroma(persist_directory=persistent_directory, embedding_function= embeddings_model)

        # retrieve the most relevent chunks from the database for the user query
        retriver = db.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={"k": 2, "score_threshold": 0.1},  # Lowered threshold to 0.1
            search_type="mmr",  # Change to MMR (Maximal Marginal Relevance)
            search_kwargs={
                "k": 2,  # Increase number of results
                "fetch_k": 20,  # Fetch more documents initially
                "lambda_mult": 0.5  # Balance between relevance and diversity
            }
        )

        relevennt_chunks = retriver.invoke(query)

        # Display the relevant results with metadata
        if len(relevennt_chunks) > 0:
            for i, chunk in enumerate(relevennt_chunks, 1):

                modified_prompt = f"""Here is the User Query/topic name: {query}
                And Here are the relevant Information.
                Do not use other info other than the additional information I provided to you.
                Relevant/Additional information:
                {' '.join([chunk.page_content for chunk in relevennt_chunks])}
                If you could not found relevant information based on user query then return message "No information found."
                """

            # make prompt template
            prompt_template = ChatPromptTemplate([
                ("system", "You are a chatbot that can answer questions based on custom stored chunk data."),
                ("system", "You should use the information in the relevant documents to answer the questions or topics."),
                ("user", modified_prompt)
                ])

            # final prompt
            final_modified_prompt = prompt_template.format_messages(query=query)

            # create the chat model and generate response
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=SECRET_KEY)
            response = llm.invoke(final_modified_prompt)

            #model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

            return ({
                'code': 200,
                'success': True,
                'message':'successfully done',
                'data': {
                    "content": response.content
                }
            })
        else:
            return ({
                'code': 200,
                'success': True,
                'message':'successfully done',
                'data': {
                    "content": ""
                }
            })

    except Exception as e:
        # Catch any errors and return a meaningful message
        return {
                'code': 401,
                'success': False,
                'message': str(e),
            }
    
