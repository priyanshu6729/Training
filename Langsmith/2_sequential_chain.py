from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "langsmith-SEQUENTIAL-CHAIN"
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv("GROQ_API_KEY"))

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
  'tags': ['SEQUENTIAL-CHAIN', 'Langsmith', 'Langchain', 'Groq'],
  'metadata': {'author': 'Priyanshu Raj', 'version': '1.0', 'description': 'A sequential chain that generates a detailed report and then summarizes it into 5 points.', 'model': 'llama-3.1-8b-instant'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
