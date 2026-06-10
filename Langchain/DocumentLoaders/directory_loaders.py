from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
loader = DirectoryLoader(path='papers', glob="*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()

print(docs[0].page_content)