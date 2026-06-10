from abc import abstractmethod
import random

class Runnable:
    def __init__(self):
        print('Runnable created!')
    
    @abstractmethod
    def invoke(self, input):
        pass

class DemoLLM(Runnable):
  def __init__(self):
    print('LLM created!')
  
  def invoke(self, prompt):
    responses = [
      "The capital of France is Paris.",
      "The capital of France is Lyon.",
      "The capital of France is Marseille."
    ]
    return {'response': random.choice(responses)}
  
  def predict(self, prompt):
    responses = [
        "The capital of France is Paris.",
        "The capital of France is Lyon.",
        "The capital of France is Marseille."
    ]
    print("this method will be decrepeted in future, use invoke instead")
    return {'response': random.choice(responses)}
  
class StrOutputParser(Runnable):
    def __init__(self):
        print('StrOutputParser created!')
    
    def invoke(self, input_dict):
        return input_dict['response']

class RunnableConnectors(Runnable):
    def __init__(self, runnable_list):
        super().__init__()
        self.runnable_list = runnable_list
        print('RunnableConnector created!')
    
    def invoke(self, input):
        for runnable in self.runnable_list:
            input = runnable.invoke(input)
        return input
   

class demoPrompt:
    def __init__(self,template,input_variables):
        print('Prompt created!')
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

template = demoPrompt(
      template="What is the capital of {country}?",
      input_variables=["country"]
    )

template1 = demoPrompt(
      template="Give me a joke about {topic}.",
      input_variables=["cat"]
    )

template2 = demoPrompt(
      template="Give me explanation for the joke: {text}.",
      input_variables=["text"]
    )

llm = DemoLLM()
parser = StrOutputParser()
chain1 = RunnableConnectors([template1, llm , parser]) 
result1 = chain1.invoke({"topic":"France"})
chain2 = RunnableConnectors([template2, llm , parser])
result = chain2.invoke({'text':result1})
print(result)