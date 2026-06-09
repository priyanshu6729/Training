from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal 
from dotenv import load_dotenv

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "return_full_text": False,
        "do_sample": False,      
        "max_new_tokens": 50   
    }
)

model = ChatHuggingFace(llm=pipe)

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the review")

parser1 = PydanticOutputParser(pydantic_object=feedback)


prompt = PromptTemplate(
    template="""
Classify the sentiment of the following review.

Rules:
- Output only one word.
- Either positive or negative.
- No explanation.
- No punctuation.

Review:
{topic}

Sentiment:
""",
    input_variables=["topic"]
)

prompt1 = PromptTemplate(
  template='write an appropriate message for the positive feedback. \n {feedback}',
  input_variables=['feedback']
)

prompt2 = PromptTemplate(
  template='write an appropriate message for the negative feedback. \n {feedback}',
  input_variables=['feedback']
)

classifier_chain = prompt | model | StrOutputParser()

branch = RunnableBranch(
    (
        lambda x: "positive" in x.lower(),
        RunnableLambda(lambda x: {"feedback": x})
        | prompt1
        | model
        | StrOutputParser()
    ),
    (
        lambda x: "negative" in x.lower(),
        RunnableLambda(lambda x: {"feedback": x})
        | prompt2
        | model
        | StrOutputParser()
    ),
    RunnableLambda(lambda x: "Invalid Sentiment")
)

chain = classifier_chain | branch

result = chain.invoke({'topic':'The product is really good and I am happy with it.'})
print(result)
chain.get_graph().print_ascii()