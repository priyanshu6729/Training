from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv("GROQ_API_KEY"))

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_model(state: ChatState) -> ChatState:
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages': [response]}


checkpointer = InMemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_model', chat_model)
graph.add_edge(START, 'chat_model')
graph.add_edge('chat_model', END)

chatbot = graph.compile(checkpointer=checkpointer)