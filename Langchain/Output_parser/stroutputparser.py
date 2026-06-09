from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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

template1 = PromptTemplate(
  template = "Write a detailed notes on {topic}",
  input_variables = ['topic']
)

template2 = PromptTemplate(
  template = "Write a 4 line summary on the following text. \n {text}",
  input_variables = ['text']
)

parser = StrOutputParser()

chain = (
    template1 
    | model 
    | parser 
    | (lambda output: {"text": output}) 
    | template2 
    | model 
    | parser
)

result = chain.invoke({'topic':'Artificial Intelligence'})
print(result)