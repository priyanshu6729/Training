from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

Text = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def multiply(self, a, b):
        return a * b
"""

splitter = RecursiveCharacterTextSplitter.from_language(
  language=Language.PYTHON,
  chunk_size = 100,
  chunk_overlap = 20
)

result = splitter.split_text(Text)
print(result)

