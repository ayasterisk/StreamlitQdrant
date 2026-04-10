import os
import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI

COLLECTION_NAME = "hotpot_qa"

@st.cache_resource
def get_resources():
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if st.secrets.get("LANGCHAIN_TRACING_V2") else "false"
    os.environ["LANGCHAIN_API_KEY"] = str(st.secrets.get("LANGCHAIN_API_KEY", ""))
    os.environ["LANGCHAIN_PROJECT"] = str(st.secrets.get("LANGCHAIN_PROJECT", "Multihop-RAG"))

    client = QdrantClient(
        url=str(st.secrets["QDRANT_URL"]),
        api_key=str(st.secrets["QDRANT_API_KEY"])
    )

    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    # ds_api_key = str(st.secrets["DEEPSEEK_API_KEY"]).strip()
    # ds_base_url = "https://api.deepseek.com/v1"

    # langchain_llm = ChatOpenAI(
    #     model='deepseek-chat',
    #     openai_api_key=ds_api_key,
    #     openai_api_base=ds_base_url,
    #     max_retries=3,
    #     temperature=0,
    #     streaming=True
    # )
    gemini_api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    
    langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=gemini_api_key,
    temperature=0,
    max_retries=3
    )

    return client, dense_model, sparse_model, langchain_llm