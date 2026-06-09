from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage , SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = HuggingFaceEndpoint(
  repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  task = "text-generation",
  huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

messages = [
  SystemMessage(content="You are a helpful assistant."),
  HumanMessage(content="What is the capital of France?")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)
