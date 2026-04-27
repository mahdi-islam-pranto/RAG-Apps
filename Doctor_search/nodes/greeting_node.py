from utilities.states.input_state import ChatState
from langchain_core.messages import AIMessage
from utilities.messages.messages import all_messages
from service.store_conversation import store_conversation


# define functions
def greeting_node(state: ChatState):
    get_greeting_message = all_messages["greeting_message_ai"]
    greeting_message = AIMessage(content=get_greeting_message)
    print("greeting node executed")

    thread_id = state["thread_id"]
    user_id = state["user_id"]

    # store conversation in the database
    data = {
        "thread_id": thread_id,
        "user_id": user_id,
        "message": greeting_message.content,
        "stage_id": '1'
    }

    # print("data: ", data)
    # store conversation in the database
    try:
        store_conversation(data)
    except Exception as e:
        print(f"An error occurred while storing conversation: {e}")
    

    return {
        "messages": [greeting_message],
        "track_stage": "rag"
    }