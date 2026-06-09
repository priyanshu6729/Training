from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from transformers import pipeline
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    task="text-generation",
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    pipeline_kwargs={
        "max_new_tokens": 256,
        "return_full_text": False,
        'do_sample': False,
    }
)

model = ChatHuggingFace(llm=pipe)

#schema
json_schema = {
  "title": "ReviewAnalysis",
  "description": "A JSON schema for analyzing reviews.",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "A brief summary of the review."
    },
    "category": {
      "type": "string",
      "description": "The category of the review, either 'movie', 'book', 'product', etc."
    },
    "sentiment": {
      "type": "string",
      "description": "The overall sentiment of the review, either 'positive' or 'negative'."
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "A list of positive aspects mentioned in the review."
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "A list of negative aspects mentioned in the review."
    }
  },
  "required": ["summary", "category", "sentiment"]
}

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Analyze the review. Return JSON matching this schema: {schema}\n{format_instructions}\nReview: {query}\nJSON:",
    input_variables=["query"],
    partial_variables={
        "schema": json_schema, 
        "format_instructions": parser.get_format_instructions()
    },
)

chain = prompt | model | parser

query = ("""The movie was fantastic! I loved it. It had great acting and a compelling story. I would highly recommend it to anyone who enjoys a good film. Specially the ending was very satisfying and left me wanting more. Overall, it was an amazing experience and I can't wait to watch it again!                               
""")

try:
    result = chain.invoke({"query": query})
    print(f"Summary: {result.get('summary')}")
    print(f"Sentiment: {result.get('sentiment')}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Chain execution completed.")
