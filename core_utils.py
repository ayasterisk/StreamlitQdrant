import os
import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI
from langchain_openai import ChatOpenAI

@st.cache_resource
def get_resources():
    # Access environment variables for LangSmith and Qdrant
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

    # Access environment variables for Qdrant
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    
    # Embedding Models
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    # DeepSeek Client
    raw_llm = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )
    
    # LangChain LLM Wrapper
    langchain_llm = ChatOpenAI(
        model='deepseek-reasoner', 
        openai_api_key=st.secrets["DEEPSEEK_API_KEY"], 
        openai_api_base="https://api.deepseek.com",
        streaming=True
    )
    
    return client, dense_model, sparse_model, raw_llm, langchain_llm

COLLECTION_NAME = "hotpot_qa"