from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions = 32)

vector = embedding.embed_query("What is the capital of France?")
'''
for embedding large data, we can use embed_documents instead of embed_query. It will return a list of vectors for each document in the list.
--------------------------------------------
embedding.embed_documents(["What is the capital of France?", "What is the capital of Germany?"])
'''

print(str(vector))