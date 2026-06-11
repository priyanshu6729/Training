from langchain.tools import tool  

@tool
def get_SubjectTopic(subject: str):
    """Returns the main topics covered in a given subject."""
    subject_topics = {
        "math": "algebra, calculus, geometry, trigonometry, statistics",
        "science": "physics, chemistry, biology, astronomy, earth science",
        "history": "ancient history, medieval history, modern history, world history",
        "literature": "poetry, novels, drama, literary analysis",
        "computer science": "programming, algorithms, data structures, machine learning"
    }
    return subject_topics.get(subject.lower(), "general knowledge")

@tool 
def get_topic_subtopics(topic: str):
    """Returns the subtopics covered in a given topic."""
    topic_subtopics = {
        "algebra": "linear equations, quadratic equations, polynomials, factoring",
        "calculus": "limits, derivatives, integrals, series",
        "geometry": "points, lines, angles, shapes, theorems",
        "trigonometry": "sine, cosine, tangent, identities, applications",
        "statistics": "mean, median, mode, variance, probability"
    }
    return topic_subtopics.get(topic.lower(), "general knowledge")

class SubjectToolkit:
    """A toolkit for retrieving information about subjects and topics."""
    
    def get_tools(self):
        return get_SubjectTopic, get_topic_subtopics

toolkit = SubjectToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(f"Tool Name: {tool.name}")
    print(f"Description: {tool.description}")

result = get_SubjectTopic.invoke("science")
print(result)
