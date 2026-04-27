from utilities.states.input_state import ChatState
from langchain_core.messages import AIMessage
from utilities.messages.messages import all_messages
from utilities.validations.dob_validation import dob_validation
from service.store_conversation import store_conversation

def date_of_birth(state: ChatState):
    # get user input from chatbot
    user_input = state["messages"][-1].content

    # store user message in the database
    thread_id = state["thread_id"]
    user_id = state["user_id"]
    data = {
        "thread_id": thread_id,
        "user_id": user_id,
        "message": user_input,
        "stage_id": '4'
    }
    try:
        store_conversation(data)
    except Exception as e:
        print(f"An error occurred while storing conversation: {e}")

    # extract the date of birth with llm if user provide other info with date of birth
    user_dob = dob_validation(user_input)
    # get invalid date of birth message from messages.py
    invalid_dob_message = all_messages["invalid_dob_message_ai"]
    invalid_dob_message = AIMessage(content=invalid_dob_message)
    # check if date of birth is valid
    if user_dob == False:
        # store invalid date of birth message in the database
        data = {
            "thread_id": thread_id,
            "user_id": user_id,
            "message": invalid_dob_message.content,
            "stage_id": '4'
        }
        try:
            store_conversation(data)
        except Exception as e:
            print(f"An error occurred while storing conversation: {e}")
        return {
            "track_stage": "3",
            "messages": [invalid_dob_message]
        }
    
    else:
        # put the date of birth in the state
        state["dob"] = user_dob
        print("Date of birth extracted: ", user_dob)
        get_dob_ai_message = all_messages["date_of_birth_ai"]
        acknowledge_message = AIMessage(
            content=get_dob_ai_message
        )

        # store acknowledge message in the database
        data = {
            "thread_id": thread_id,
            "user_id": user_id,
            "message": acknowledge_message.content,
            "stage_id": '4'
        }
        try:
            store_conversation(data)
        except Exception as e:
            print(f"An error occurred while storing conversation: {e}")
            
        print("date of birth node executed")
        return {
            "messages": [acknowledge_message],
            "track_stage": "4",
            "dob": user_dob
        }