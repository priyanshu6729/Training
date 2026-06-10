from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence , RunnableLambda , RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=pipe)
parser = StrOutputParser()

prompt = PromptTemplate(
  template="Write 5 lines about {topic}",
  input_variables=['topic']
)

chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(chain, parallel_chain)
result = final_chain.invoke({'topic': 'programming'})
print(result)
