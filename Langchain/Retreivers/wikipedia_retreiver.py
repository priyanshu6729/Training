from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=5,
    lang="en"
)

try:
    docs = retriever.invoke("Virat Kohli")

    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:300])

except Exception as e:
    print("Error:", e)