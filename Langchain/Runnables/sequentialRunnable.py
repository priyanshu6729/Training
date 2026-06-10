from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
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

prompt = PromptTemplate(
  template="Write a joke about {topic}",
  input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)

print(chain.invoke({'topic': 'programming'}))