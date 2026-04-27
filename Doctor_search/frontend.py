import streamlit as st
from agent_backend import chatbot, thread_id, user_id
from langchain_core.messages import HumanMessage, SystemMessage

st.title("Ask me anything")

# initialize message history with session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# sample message history
# st.session_state["message_history"] = [
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hello, how can I help you?"},
# ]

# get user input
user_input = st.chat_input("Type here: ")

# show message history in the chat interface
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# very first invoke for showing greeting on first load
if "initialized" not in st.session_state:
    response = chatbot.invoke(

        {"messages": [], 
        "track_stage": "",
        "thread_id": thread_id,
        "user_id": user_id
         },
        config={"configurable": {"thread_id": thread_id}}
    )

    print("first invoke")

    ai_message = response["messages"][-1].content
    st.session_state["message_history"] = [
        {"role": "assistant", "content": ai_message}
    ]
    st.session_state["initialized"] = True

    # show the AI message in the interface
    with st.chat_message("assistant"):
        st.text(ai_message)


# show user input in the chat interface
if user_input:
    # put user input in the message history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.text(user_input)


    # thread_id = "1"

    config = {'configurable': {"thread_id": thread_id}}

    # get chatbot response
    response = chatbot.invoke({"messages": [SystemMessage(content="You are a helpful chatbot."),
                                  HumanMessage(content=user_input)]},config=config)
    ai_response = response['messages'][-1].content
    # put ai response in the message history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
    # show chatbot response in the chat interface
    with st.chat_message('assistant'):
        st.text(ai_response)

    # show checkpointer memory
    with st.expander("LangGraph Memory"):
        state = chatbot.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        st.json(state.values)