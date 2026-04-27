from utilities.states.input_state import ChatState

# define tracking node
def tracking_node(state: ChatState):
    if state["track_stage"] == "":
        return {"track_stage": "0"}
    else:
        print("track stage: ", state["track_stage"])
        pass