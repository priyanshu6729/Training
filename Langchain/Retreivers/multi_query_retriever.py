from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
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
    )
]

vectorstore = FAISS.from_documents(
  documents=documents,
  embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False
    }
)
model = ChatHuggingFace(llm=pipe)

multiquery = MultiQueryRetriever.from_llm(
  retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
  llm = model
)

query = "Best of Cricket history?"

results = multiquery.invoke(query)

for result in results:
    print(result.metadata)
    print(result.page_content)