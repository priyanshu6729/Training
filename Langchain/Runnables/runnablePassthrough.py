from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence , RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=pipe)
parser = StrOutputParser()

prompt1 = PromptTemplate(
  template="Write a joke about {topic}",
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template="Explain the joke: {joke}",
  input_variables=['joke']
)

result1 = RunnableSequence(prompt1, model, parser)

parallelChain = RunnableParallel({
   'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final = RunnableSequence(result1,parallelChain)

finally_result = final.invoke({'topic': 'programming'})
print(finally_result)

