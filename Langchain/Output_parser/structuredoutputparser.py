from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
  ResponseSchema(name="fact 1", description="First fact about the topic"),
  ResponseSchema(name="fact 2", description="second fact about the topic"),
  ResponseSchema(name="fact 3", description="third fact about the topic"),
  ResponseSchema(name="fact 4", description="fourth fact about the topic"),
  ResponseSchema(name="fact 5", description="fifth fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = ChatPromptTemplate.from_messages(
  [
    {
      "template": 'Extract 5 facts about the {topic} \n {format_instructions}',
      "input_variables": ['topic']
    }
  ]
)

template = template.partial(format_instructions=parser.get_format_instructions())

chain = template | model | parser

result = chain.invoke({'topic':'Python programming language'})
print(result)
print(type(result))