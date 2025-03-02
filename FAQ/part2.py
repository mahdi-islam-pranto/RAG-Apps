import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
load_dotenv()


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

# Get the user query/promt text
user_query = "মানিক কে?"


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

    # put the relevant chunks + user query into the LLM model openai and generate the answer 
    modified_prompt = f"""Here is the User Query: {user_query}
    And Here are the relevant Information where you can find the answer.
    Do not use other info other than the additional information I provided to you.
    Relevant information:
    {' '.join([chunk.page_content for chunk in relevennt_chunks])}
    """

    # create the chat model and generate response
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(modified_prompt)
    print("\n--- Answer ---")
    print(response.content)
else:
    print("No relevant documents found for the query.")



