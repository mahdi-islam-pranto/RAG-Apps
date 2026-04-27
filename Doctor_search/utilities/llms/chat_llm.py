from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# define llm
chat_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")