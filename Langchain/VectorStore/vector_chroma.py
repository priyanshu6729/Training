from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(
        page_content="""
Cricket is a bat-and-ball sport played between two teams of eleven players.
The game originated in England and is now one of the most popular sports in countries like India, Australia, and England.
International formats include Test, ODI, and T20 cricket.
        """,
        metadata={
            "source": "cricket_basics.txt",
            "category": "Introduction",
            "page": 1
        }
    ),

    Document(
        page_content="""
Sachin Tendulkar is regarded as one of the greatest batsmen in cricket history.
He scored 100 international centuries and accumulated more than 34,000 international runs during his career.
He represented India from 1989 to 2013.
        """,
        metadata={
            "source": "players.txt",
            "player": "Sachin Tendulkar",
            "country": "India"
        }
    ),

    Document(
        page_content="""
The ICC Cricket World Cup is the premier One Day International cricket tournament.
The tournament is organized by the International Cricket Council (ICC) every four years.
India won the World Cup in 1983, 2011, and 2023.
        """,
        metadata={
            "source": "world_cup.txt",
            "tournament": "ICC Cricket World Cup"
        }
    ),

    Document(
        page_content="""
The Indian Premier League (IPL) is a professional T20 cricket league in India.
It was founded in 2008 and features franchise teams from various Indian cities.
The IPL is among the most valuable sporting leagues in the world.
        """,
        metadata={
            "source": "ipl.txt",
            "league": "IPL",
            "format": "T20"
        }
    ),

    Document(
        page_content="""
Virat Kohli is known for his consistency across all formats of cricket.
He has captained the Indian cricket team and holds numerous batting records.
Kohli is considered one of the best modern-day batsmen.
        """,
        metadata={
            "source": "players.txt",
            "player": "Virat Kohli",
            "country": "India"
        }
    )
]

vectorStore = Chroma(
  embedding_function=HuggingFaceEmbeddings(),
  collection_name="cricket_docs",
  persist_directory="./chroma_db"
)

vectorStore.add_documents(documents)
print("Documents added to Chroma vector store successfully.")

data = vectorStore.get(include=['documents','embeddings', 'metadatas'])
print("Retrieved data from Chroma vector store:")
for doc, embedding, metadata in zip(data['documents'], data['embeddings'], data['metadatas']):
    print(f"Document: {doc}")
    print(f"Embedding: {embedding[:5]}...")
    print(f"Metadata: {metadata}")
    print("-" * 50)


UpdateDoc = Document(
        page_content="""MS Dhoni is a former Indian cricketer and captain of the Indian national team.
He is known for his calm demeanor and exceptional wicket-keeping skills.
Dhoni led India to victory in the 2007 ICC World Twenty20, 2011 ICC Cricket World Cup, and 2013 ICC Champions Trophy.""",
        metadata={
            "source": "players.txt",
            "player": "MS Dhoni",
            "country": "India"
        }
    )

vectorStore.add_documents([UpdateDoc])
print("Document updated successfully.")

similar = vectorStore.similarity_search_with_score(
    query="Who is the best batsman in cricket?",
    k=2
)

print("Similarity search results:")

for doc, score in similar:
    print(f"Document: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print("-" * 50)