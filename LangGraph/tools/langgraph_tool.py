from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun  # kept as-is; suppress warning below
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import yfinance as yf  # replaces raw Yahoo Finance v7 API (which now returns 401)
import warnings
import os

# Suppress the langchain-community deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv("GROQ_API_KEY"))

search_tool = DuckDuckGoSearchRun()

@tool
def calculator(num1: float, num2: float, operation: str) -> float:
    """
    A simple calculator tool that performs basic arithmetic operations.
    """
    if operation == 'add':
        return num1 + num2
    elif operation == 'subtract':
        return num1 - num2
    elif operation == 'multiply':
        return num1 * num2
    elif operation == 'divide':
        if num2 == 0:
            raise ValueError("Cannot divide by zero.")
        return num1 / num2
    else:
        raise ValueError("Invalid operation. Choose from 'add', 'subtract', 'multiply', or 'divide'.")

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current (latest available) stock price for a given ticker symbol.
    Uses yfinance under the hood — no API key required.
    Returns a human-readable string with the price and currency.
    """
    ticker = ticker.upper().strip()
    stock = yf.Ticker(ticker)

    # fast_info is the lightest call — just the last price
    price = stock.fast_info.get("last_price")

    if price is None:
        # Fallback: pull the most recent closing price from 1-day history
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError(
                f"Could not retrieve stock price for ticker: {ticker}. "
                "Please verify the symbol is correct (e.g. 'AAPL', 'TSLA', 'MSFT')."
            )
        price = float(hist["Close"].iloc[-1])

    currency = stock.fast_info.get("currency", "USD")
    return f"{ticker}: {price:.2f} {currency}"

# Tool list
tools = [search_tool, calculator, get_stock_price]

# Make LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_model(state: ChatState) -> ChatState:
    """LLM node — invokes the model with the current message history."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node('chat_model', chat_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_model")
graph.add_conditional_edges(
    "chat_model",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge("tools", "chat_model")

chat = graph.compile()

if __name__ == "__main__":
    output = chat.invoke({'messages': [HumanMessage(content="What is the result: 2*3?")]})
    print(output['messages'][-1].content)