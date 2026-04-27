from utilities.states.input_state import ChatState
from service.rag.doctor_search import documents, langchain_documents, vectorstore
from service.rag.search.hybrid import hybrid_langchain_retriever
from utilities.llms.chat_llm import chat_llm
from utilities.prompts.doctor_rag_search import doctor_rag_search
from langchain_core.messages import HumanMessage
import os

def rag_node(state: ChatState):
    print("rag node executed")
    # get user input from chatbot
    user_input = state["messages"][-1].content
    print("user input: ", user_input)

    # extract the name of the doctor or department from the user input using llm
    llm_prompt_nd = f"""
        {doctor_rag_search["doctor_extraction_prompt"]}
        User input: {user_input}"""

    llm_response_nd = chat_llm.invoke([HumanMessage(content=llm_prompt_nd)])

    # check if the llm response is unknown
    if llm_response_nd.content == "Unknown":
        print("llm response: ", llm_response_nd.content)
        return {
            "messages": ["Please provide the name of the doctor or department you want to book an appointment with"],
            "track_stage": "rag"
        }
    
    else:
        extracted_name = llm_response_nd.content
        

    # get the similar documents using hybrid retriever
    result = hybrid_langchain_retriever(
        query=extracted_name,
        lc_documents=langchain_documents,
        vectorstore=vectorstore
    )

    print(f"retrieved documents: {result}")

    # llm prompt
    llm_prompt_doctor_verification = f"""
    You are a medical appointment booking chatbot. You are given a doctor name or department name and a list of documents (all related doctor name and department name, etc.). The user input is: {user_input}. The user can say the name of the doctor or the department of the doctor. We have the found the following doctors and departments in our hospital: {result}. Verify and Tell user about the doctor or the department he is asking for. If our hospital does not have the doctor or the department he is asking for, say that we do not have the doctor or the department he is asking for.
"""
    llm_response = chat_llm.invoke([HumanMessage(content=llm_prompt_doctor_verification)])

    doctor_verification_result = llm_response.content

    # show sample message
    # sample_message = "I got your message. I am a rag node. I will do rag things here. tell me your name"
    # print("sample message: ", sample_message)
    # return the message
    return {
        "messages": [doctor_verification_result],
        "track_stage": "1"
    }