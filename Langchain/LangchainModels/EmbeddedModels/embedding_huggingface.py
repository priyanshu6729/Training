from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = ["the capital of France is paris.", "What is the capital of Germany?", "What is the capital of Italy?"]

query = "Where is paris located?"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Most similar document: {documents[index]} with score {score}")