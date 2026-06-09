from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 5 facts about the topic. Return ONLY a valid JSON list of strings. Do not include any other text.\n{format_instructions}"),
    ("human", "{topic}")
])

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

result = chain.invoke({'topic':'Python programming language'})

print(result)
print(type(result))