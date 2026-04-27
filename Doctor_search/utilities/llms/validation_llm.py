from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()



# name validation llm
def validation_llm(user_input, validation_prompt):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    merged_prompt = validation_prompt + f" Now Here is the user input: {user_input}"
    response = llm.invoke([HumanMessage(content=merged_prompt)])
    return response.content

# test the name_llm
# print(name_llm("I am amzad hossain pranto"))
