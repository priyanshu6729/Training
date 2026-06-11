from pydantic import BaseModel, Field
from langchain_community.tools import StructuredTool

class CalculatorInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

def add(a: int, b: int):
    return a + b

tool = StructuredTool.from_function(
    func=add,
    name="add",
    description="Add two numbers",
    args_schema=CalculatorInput
)

print(tool.invoke({"a": 10, "b": 20}))