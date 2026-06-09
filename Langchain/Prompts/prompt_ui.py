from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.header("Chat Model")

role_input = st.selectbox("Select a role", ["assistant", "researcher", "student"])
style_input = st.selectbox("Select a style", ["Formal", "Informal", "Technical"])
length_input = st.slider("Select response length", min_value=50, max_value=500, value=100, step=50)

prompt = load_prompt("template.json")

if st.button("Generate Response"):

    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    chat_model = ChatHuggingFace(llm=llm)

    with st.spinner("Generating response..."):
        response = chat_model.invoke(prompt.format(role_input=role_input, style_input=style_input, length_input=length_input))
        st.success("Response generated!")
        st.write(response.content)