from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

pipe = HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs= {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=pipe)

template = PromptTemplate(
  template = "Generate a summary from the following text:\n\n{text}.",
  input_variables = ["text"]
)

parser = StrOutputParser()

loader = TextLoader("example.txt", encoding="utf-8")
documents = loader.load()

print(documents[0].page_content)

chain = template | model | parser

result = chain.invoke({"text": documents[0].page_content})

print("Summary ---------------------------------------------->>>")
print(result)
