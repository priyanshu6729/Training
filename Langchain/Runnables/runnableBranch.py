from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence , RunnableLambda , RunnablePassthrough , RunnableBranch
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

prompt1 = PromptTemplate(
  template="summary the text {text}",
  input_variables=['text']
)

chain = RunnableSequence(prompt, model, parser)

summary_chain = (
    RunnableLambda(lambda x: {"text": x})
    | prompt1
    | model
    | parser
)

branch = RunnableBranch(
    (lambda x: len(x.split()) > 300 , summary_chain ),
    RunnablePassthrough()
)

final_chain = RunnableSequence(chain, branch)
result = final_chain.invoke({'topic': 'programming'})
print(result)
