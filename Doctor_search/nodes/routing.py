from utilities.states.input_state import ChatState

# define routing node
def routing_node(state: ChatState):
    if state["track_stage"] == "0":
        return "greeting"
    elif state["track_stage"] == "rag":
        return "rag_node"
    elif state["track_stage"] == "1":
        return "name"
    elif state["track_stage"] == "2":
        return "age"
    elif state["track_stage"] == "3":
        return "date_of_birth"
    
    return "chat_node"