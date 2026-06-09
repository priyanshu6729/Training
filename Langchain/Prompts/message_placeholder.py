from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('chat_history.txt') as f:
      chat_history.extend(f.readlines())

prompt = chat_template.invoke({'chat_history': chat_history , 'query': 'What is the status of my order?'})

print(prompt)