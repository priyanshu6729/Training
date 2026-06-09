from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False
    },
)

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

chat_model = ChatHuggingFace(llm=model)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break
    
    chat_history.append(HumanMessage(content=user_input))

    response = chat_model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=response.content))
    print(f"Chatbot: {response.content}")