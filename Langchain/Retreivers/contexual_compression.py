from langchain_community.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

docs = [
  Document(
    page_content="""LangChain is a powerful framework for building applications with language models. It provides tools and abstractions to make it easier to work with LLMs, including prompt management, memory, and more.""",
    metadata={'source': 'langchain.com'}
  ),
  Document(
    page_content="""Contextual compression is a technique used to reduce the size of input data while preserving its meaning. It can be useful for improving the efficiency of language models by allowing them to process larger contexts without running into token limits.""",
    metadata={'source': 'example.com'}
  ),
  Document(
    page_content="""The Hugging Face Transformers library provides a wide range of pre-trained models for various NLP tasks, including text generation, classification, and more. It also offers tools for fine-tuning models on custom datasets.""",
    metadata={'source': 'huggingface.co'}
  ),
  Document(
    page_content="""The history of Cinema is a fascinating journey that spans over a century. It began in the late 19th century with the invention of motion picture cameras and has evolved into a global industry that produces a wide variety of films, from silent movies to modern blockbusters.""",
    metadata={'source': 'cinemahistory.com'}
  )
]

embedding_model = HuggingFacePipeline.from_model_id(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    pipeline_kwargs={"truncation": True, "padding": True}
)
vectorstore = FAISS.from_documents(docs, embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "do_sample": False
    }
)
llm = ChatHuggingFace(llm=pipe)
compressor = LLMChainExtractor.from_llm(llm=llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

query = "What is contextual compression and how does it relate to language models?"
results = compression_retriever.invoke(query)

for result in results:
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print("-" * 50)
