import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

# fast api
from fastapi import FastAPI , File, HTTPException, UploadFile, Form
from pydantic import BaseModel

app = FastAPI()


@app.post("/response")
async def get_response(user_query: str = Form):
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # The path to the database directory
        persistent_directory = os.path.join(current_dir, "db", "chroma_db")

        # embedding model
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
            )

        # load the existing vector store with the embeddings
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function= embedding_model
            )

        # # Get the user query/promt text
        # user_query = "Query: tell me about the graphics of the book The Gale Encyclopedia of Medicine 2?"


        # retrieve the most relevent chunks from the database for the user query
        retriver = db.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={"k": 2, "score_threshold": 0.1},  # Lowered threshold to 0.1
            search_type="mmr",  # Change to MMR (Maximal Marginal Relevance)
            search_kwargs={
                "k": 4,  # Increase number of results
                "fetch_k": 20,  # Fetch more documents initially
                "lambda_mult": 0.5  # Balance between relevance and diversity
            }
        )


        relevennt_chunks = retriver.invoke(user_query)

        # Display the relevant results with metadata
        print("\n--- Relevant Documents ---")
        if len(relevennt_chunks) > 0:
            for i, chunk in enumerate(relevennt_chunks, 1):
                print(f"chunk:{i} {chunk.page_content}\n")

            modified_prompt = f"""Here is the User Query/topic name: {user_query}
            And Here are the relevant Information where you can find what to tell to the customer.
            Do not use other info other than the additional information I provided to you.
            Relevant/Additional information:
            {' '.join([chunk.page_content for chunk in relevennt_chunks])}
            """

            

            # make prompt template
            prompt_template = ChatPromptTemplate([
                ("system", "You are a chatbot that can answer questions and provide suggetions for a call center agent about what to tell a customer."),
                ("system", "If the agent only type topic of the question, you should provide what to tell the customer based on the relevant documents."),
                ("system", "You should use the information in the relevant documents to answer the questions or topics."),
                ("user", modified_prompt)
                ])

            # final prompt
            final_modified_prompt = prompt_template.format_messages(user_query=user_query)
            # put the relevant chunks + user query into the LLM model openai and generate the answer 
            
            # create the chat model and generate response
            # llm = ChatOpenAI(model="gpt-4o")
            llm = ChatDeepSeek(model="deepseek-chat")
            response = llm.invoke(final_modified_prompt)
            print("\n--- Prompt ---")
            print(final_modified_prompt)
            print("\n--- Answer ---")
            print(response.content)

            # save the response to a file with UTF-8 encoding
            with open("AIresponse.txt", "w", encoding="utf-8") as f:
                f.write(response.content)
        else:
            print("No relevant documents found for the query.")

    except Exception as e:
        return {"error": str(e)}
    
    return {
        "status": "success",
        "user_query": user_query,
        "final_prompt": final_modified_prompt,
        "AI_response": response.content,
        "relevennt_chunks": [chunk.page_content for chunk in relevennt_chunks],
        "number_of_relevennt_chunks": len(relevennt_chunks)
        }





