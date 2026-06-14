import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

thread_id = '1'
CONFIG = {'configurable': {'thread_id': thread_id}}

st.set_page_config(
    page_title="Assistant Chatbot",
    page_icon="🤖",
)

st.markdown(
    "<h1 style='text-align:center;'>🤖 Assistant Chatbot</h1>",
    unsafe_allow_html=True
)

if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]},config=CONFIG)
    st.session_state['message_history'].append({"role": "assistant", "content": response['messages'][-1].content})
    with st.chat_message('assistant'):
        st.markdown(response['messages'][-1].content)