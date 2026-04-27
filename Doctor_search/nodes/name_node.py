from utilities.states.input_state import ChatState
from langchain_core.messages import AIMessage
from utilities.messages.messages import all_messages
from utilities.validations.name_validation import name_validation
from service.store_conversation import store_conversation

def name(state: ChatState):
    get_name_message = all_messages["name_message_ai"]
    ask_age_message = AIMessage(content=get_name_message)
    print("name node executed")

    # get user input from chatbot
    user_input = state["messages"][-1].content

    # store user message in the database
    thread_id = state["thread_id"]
    user_id = state["user_id"]
    data = {
        "thread_id": thread_id,
        "user_id": user_id,
        "message": user_input,
        "stage_id": '2'
    }

    try:
        store_conversation(data)
    except Exception as e:
        print(f"An error occurred while storing conversation: {e}")

    # extract the name with llm if user provide other info with name
    user_name = name_validation(user_input)

    # get invalid name message from messages.py
    invalid_name_message = all_messages["invalid_name_message_ai"]
    invalid_name_message = AIMessage(content=invalid_name_message)
    # check if name is valid
    if user_name == False:
        # store invalid name message in the database
        data = {
            "thread_id": thread_id,
            "user_id": user_id,
            "message": invalid_name_message.content,
            "stage_id": '2'
        }
        try:
            store_conversation(data)
        except Exception as e:
            print(f"An error occurred while storing conversation: {e}")

        # return invalid name message    
        return {
            "track_stage": "1",
            "messages": [invalid_name_message]
        }
    else:
        # put the name in the state
        state["name"] = user_name
        print("Name extracted: ", user_name)

        # db operation to insert name in to the database
        data = {
            "thread_id": thread_id,
            "user_id": user_id,
            "message": ask_age_message.content,
            "stage_id": '2'
        }
        try:
            store_conversation(data)
        except Exception as e:
            print(f"An error occurred while storing conversation: {e}")
        

        return {
            "name": user_name,
            "messages": [ask_age_message],
            "track_stage": "2"
        }
        