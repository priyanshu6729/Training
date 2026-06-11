from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def currency_conversion(amount: float, from_currency: str, to_currency: str) -> float:
    """Converts a given amount from one currency to another currency"""
    currency = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.75,
        "JPY": 110.0,
        "CNY": 6.5
    }
    return amount * currency.get(to_currency.upper(), 1.0) / currency.get(from_currency.upper(), 1.0)

@tool
def convert(a: int, b: int) -> int:
    """Converts two integers by adding them together."""
    return a + b

pipe = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen2.5-3B-Instruct",
  task = "text-generation",
  huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

llm = ChatHuggingFace(llm=pipe)

messages = HumanMessage(content="Convert 100 USD to EUR and also add 5 and 10 together.")

llm_with_tools = llm.bind_tools([currency_conversion, convert])

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if tool_call.tool_name == "currency_conversion":
        tool_response = currency_conversion.invoke(tool_call.args)
        messages.append(tool_response)
    elif tool_call.tool_name == "convert":
        tool_response = convert.invoke(tool_call.args)
        messages.append(tool_response)
    else:
        print(f"Unknown tool: {tool_call.tool_name}")

result = llm.invoke(messages)
print(result.content)


