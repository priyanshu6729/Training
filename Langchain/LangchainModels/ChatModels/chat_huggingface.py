# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os

# load_dotenv()

# llm = HuggingFaceEndpoint(
#   repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#   task = "text-generation",
#   huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of France?")

# print(result)
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke("Explain embeddings")

print(response)