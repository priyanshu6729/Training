from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class CustomInput(BaseModel):
    a : int = Field(description="An integer input")
  
class CustomTool(BaseTool):
    name: str = Field(required=True, description="Name of the tool")
    description: str = Field(required=True, description="Description of the tool")
    args_schema: Type[BaseModel] = CustomInput

    def fibonacci(self,n: int):
        if n == 0 or n == 1:
            return n
        else:
            return self.fibonacci(n-1) + self.fibonacci(n-2)
        
    def _run(self, a: int) -> str:
        result = self.fibonacci(a)
        return f"The {a}th Fibonacci number is: {result}"

tool = CustomTool(
    name="FibonacciTool",
    description="A tool to calculate the nth Fibonacci number",
)

result = tool.invoke({"a": 10})

print(result)
print(tool.description)


