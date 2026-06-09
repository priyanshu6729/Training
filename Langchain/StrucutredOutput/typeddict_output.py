from langchain_huggingface import ChatHuggingFace
from typing import TypedDict , Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace(model="tinyspeck/roberta-base-imdb", task="text-classification")

class Review(TypedDict):
    summary:Annotated[str, "A brief summary of the review"]
    category:Annotated[str, "The category of the review, either 'movie', 'book', 'product', etc."]
    sentiment:Annotated[str, "The overall sentiment of the review, either 'positive' or 'negative'"]
    pros:Annotated[list[str], "A list of positive aspects mentioned in the review"]
    cons:Annotated[list[str], "A list of negative aspects mentioned in the review"]
  
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The movie was fantastic! I loved it. It had great acting and a compelling story. I would highly recommend it to anyone who enjoys a good film. Specially the ending was very satisfying and left me wanting more. Overall, it was an amazing experience and I can't wait to watch it again!                               
""")

print(result)