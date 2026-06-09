from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv  

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
  model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  task = "text-generation",
  pipeline_kwargs = {
    "max_new_tokens": 256,
    "return_full_text": False,
    "do_sample": False,
  }
)

model = ChatHuggingFace(llm=pipe)

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The city where the person lives")
  

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="<|system|>\nExtract person details into JSON. Do not repeat the schema instructions. Respond ONLY with the JSON object.\n{format_instructions}</s>\n<|user|>\n{text}</s>\n<|assistant|>\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model

result = chain.invoke({'text':'John is a 25 year old software engineer living in New York.'})

print(result)