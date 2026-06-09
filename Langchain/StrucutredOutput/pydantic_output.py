from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from transformers import pipeline
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    task="text-generation",
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    pipeline_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.1,
    }
)

model = ChatHuggingFace(llm=pipe)

class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    category: str = Field(description="The category of the review, either 'movie', 'book', 'product', etc.")
    sentiment: str = Field(description="The overall sentiment of the review, either 'positive' or 'negative'")
    pros: Optional[list[str]] = Field(default=None,description="A list of positive aspects mentioned in the review")
    cons: Optional[list[str]] = Field(default=None,description="A list of negative aspects mentioned in the review")

parser = PydanticOutputParser(pydantic_object=Review)

prompt = PromptTemplate(
    template="Analyze the following review.\n{format_instructions}\nReview: {query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

query = ("""The movie was fantastic! I loved it. It had great acting and a compelling story. I would highly recommend it to anyone who enjoys a good film. Specially the ending was very satisfying and left me wanting more. Overall, it was an amazing experience and I can't wait to watch it again!                               
""")

try:
    result = chain.invoke({"query": query})
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Chain execution completed.")
