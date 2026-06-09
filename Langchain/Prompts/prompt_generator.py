from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
  template="""
You are a helpful {role_input} that generates prompts for a user based on the paper they have uploaded. The user will provide you with the title of the paper, the abstract, and the main points of the paper. Based on this information, you will generate a {style_input} prompt that the user can use to ask questions about the paper. The prompt should be {length_input}, and should include the title of the paper, the abstract, and the main points. The prompt should also include a question that the user can ask about the paper.
""",
    input_variables=["role_input", "style_input", "length_input"],
    validate_template=True
)

template.save("template.json")