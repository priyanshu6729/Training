from langchain_text_splitters import RecursiveCharacterTextSplitter , CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """
# Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# """

loader = PyPDFLoader('sample-text-only-pdf-a4-size.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 40,
  chunk_overlap = 10
)

result = splitter.split_documents(docs)
print(result)
