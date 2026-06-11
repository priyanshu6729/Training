import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


st.set_page_config(
    page_title="YouTube RAG Assistant",
    layout="wide"
)

st.title("YouTube RAG Assistant")
st.write("Ask questions about any YouTube video with transcripts.")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pipe = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "do_sample": False,
            "return_full_text": False,
        },
    )

    llm = ChatHuggingFace(llm=pipe)
    return embeddings, llm


embeddings, llm = load_models()


video_id = st.text_input(
    "Enter YouTube Video ID",
    value="C0gErQtnNFE"
)

question = st.text_input(
    "Ask a Question",
    placeholder="What is this video about?"
)


if st.button("Generate Answer"):

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

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )
            vectorstore.save_local(f"faiss_index_{video_id}")

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )

        prompt = PromptTemplate(
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

        chain = (
            {
                "context": retriever
                | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Generating answer..."):
            answer = chain.invoke(question)

        st.success("Answer Generated")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")

        docs = retriever.invoke(question)

        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content)

    except Exception as e:
        st.error(f"Error: {e}")