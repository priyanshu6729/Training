from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

search_tool = ShellTool()
result = search_tool.invoke("ls")
print(result)