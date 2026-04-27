from utilities.states.input_state import ChatState
from utilities.llms.chat_llm import chat_llm
from service.store_conversation import store_conversation


def chat_node(state: ChatState):
    llm_response = chat_llm.invoke(state["messages"])

    # store user message in the database
    thread_id = state["thread_id"]
    user_id = state["user_id"]
    data = {
        "thread_id": thread_id,
        "user_id": user_id,
        "message": llm_response.content,
        "stage_id": '5'
    }
    try:
        store_conversation(data)
    except Exception as e:
        print(f"An error occurred while storing conversation: {e}")

    return {"messages": [llm_response]}

