from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"return_full_text": False}
)

model1 = ChatHuggingFace(llm=pipe)
model2 = ChatOpenAI()

prompt1 = PromptTemplate(
  template = 'Generate 5 line about {topic}',
  input_variables= ['topic']
)

prompt2 = PromptTemplate(
  template = 'Give an example for the following text. \n {text}',
  input_variables= ['text']
)

prompt3 = PromptTemplate(
  template = 'Merge the following two texts into one. \n Text 1: {text1} \n Text 2: {text2}',
  input_variables= ['text1', 'text2']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    branches = {
        "branch1": prompt1 | model1 | parser,
        "branch2": prompt1 | model2 | parser
    })

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'topic':'climate'})
print(result)
chain.get_graph().print_ascii()