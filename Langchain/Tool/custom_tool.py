from langchain.tools import tool  

@tool
def getSubjectTopic(subject: str):
    """Returns the main topics covered in a given subject."""
    subject_topics = {
        "math": "algebra, calculus, geometry, trigonometry, statistics",
        "science": "physics, chemistry, biology, astronomy, earth science",
        "history": "ancient history, medieval history, modern history, world history",
        "literature": "poetry, novels, drama, literary analysis",
        "computer science": "programming, algorithms, data structures, machine learning"
    }
    return subject_topics.get(subject.lower(), "general knowledge")

result = getSubjectTopic.invoke("science")
print(result)