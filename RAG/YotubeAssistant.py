import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Assistant",
    layout="wide"
)

st.title("YouTube RAG Assistant")
st.write("Ask questions about any YouTube video with transcripts.")


MODEL_OPTIONS = {
    "Gemma2 9B (fast & smart)":     "gemma2-9b-it",
    "LLaMA 3.1 8B (balanced)":      "llama-3.1-8b-instant",
    "LLaMA 3.3 70B (best quality)": "llama-3.3-70b-versatile",
}

selected_model_label = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_label]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


video_id = st.text_input(
    "Enter YouTube Video ID",
    value="C0gErQtnNFE"
)

question = st.text_input(
    "Ask a Question",
    placeholder="What is this video about?"
)


if st.button("Generate Answer"):

    llm = ChatGroq(
        model=selected_model,
        api_key=os.getenv("GROQ_API_KEY")
    )

    try:
        with st.spinner("Fetching transcript..."):

            api = YouTubeTranscriptApi()

            transcript_list = api.fetch(
                video_id=video_id,
                languages=["en"]
            )

            transcript = " ".join(
                item.text for item in transcript_list
            )

        with st.spinner("Creating vector store..."):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.create_documents([transcript])

            vectorstore = FAISS.from_documents(chunks, embeddings)

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )

        docs = retriever.invoke(question)
        context = format_docs(docs)

        answer_prompt = PromptTemplate(
            template="""
You are a helpful assistant.

Use only the provided context to answer.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided context."

Answer:
""",
            input_variables=["context", "question"]
        )

        answer_chain = (
            {
                "context": RunnableLambda(lambda _: context),
                "question": RunnablePassthrough(),
            }
            | answer_prompt
            | llm
            | StrOutputParser()
        )

        summary_prompt = PromptTemplate(
            template="""
Summarize the following context in 3-4 concise sentences.

Context:
{context}

Summary:
""",
            input_variables=["context"]
        )

        summary_chain = summary_prompt | llm | StrOutputParser()

        with st.spinner("Generating answer..."):
            answer = answer_chain.invoke(question)

        with st.spinner("Summarizing context..."):
            summary = summary_chain.invoke({"context": context})

        st.success("Answer Generated")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Context Summary")
        st.write(summary)

        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content)

    except Exception as e:
        st.error(f"Error: {e}")