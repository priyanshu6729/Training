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

template = PromptTemplate(
  template = 'Generate 5 line about {topic}',
  input_variables= ['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'AI'})

print(result)

chain.get_graph().print_ascii()