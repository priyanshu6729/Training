from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"return_full_text": False}
)

model = ChatHuggingFace(llm=pipe)

prompt1 = PromptTemplate(
  template = 'Generate 5 line about {topic}',
  input_variables= ['topic']
)

prompt2 = PromptTemplate(
  template = 'Give an example for the following text. \n {text}',
  input_variables= ['text']
)

parser = StrOutputParser()

chain = (
  prompt1 | model | parser | (lambda output: {"text": output}) | prompt2 | model | parser
)

result = chain.invoke({'topic':'climate'})

print(result)
chain.get_graph().print_ascii()