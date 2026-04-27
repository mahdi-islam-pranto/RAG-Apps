
from utilities.states.input_state import ChatState
from langchain_core.messages import AIMessage
from utilities.messages.messages import all_messages

def invalid_name(state: ChatState):
    invalid_name_message = all_messages["invalid_name_message_ai"]
    ask_name_message = AIMessage(content=invalid_name_message)
    print("invalid name node executed")
    return {
        "messages": [ask_name_message],
        "track_stage": "1"
    }