from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# pipe = HuggingFacePipeline.from_model_id(
#     model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task= "text-generation",
#     pipeline_kwargs= {
#         "max_new_tokens": 512,
#         "do_sample": True,
#         "temperature": 0.7}
# )

# model = ChatHuggingFace(llm=pipe)

# prompt = PromptTemplate(
#     template = "Generate a summary from the following text:\n\n{text}.",
#     input_variables = ["text"]
# )

# parser = StrOutputParser()

loader = PyPDFLoader('sample-text-only-pdf-a4-size.pdf')
# documents = loader.load()

for doc in loader.lazy_load():
    print(doc.page_content)
    print("Metadata ---------------------------------------------->>>")
    print(doc.metadata)

# print(documents[0].page_content)
# print("Metadata ---------------------------------------------->>>")
# print(documents[0].metadata)