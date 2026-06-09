from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(models="text-embedding-3-small", dimensions=32)

documents = ["What is the capital of France?", "What is the capital of Germany?", "What is the capital of Italy?"]

query = "Where is paris located?"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1])

print(f"Query: {query}")
print(f"Most similar document: {documents[index]} with score {score}")