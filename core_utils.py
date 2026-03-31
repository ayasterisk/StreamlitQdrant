import os
import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI
from langchain_openai import ChatOpenAI

@st.cache_resource
def get_resources():
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "Multi-hop-RAG")

    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    api_key = st.secrets["DEEPSEEK_API_KEY"]
    base_url = "https://api.deepseek.com"
    
    raw_llm = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 5. LangChain LLM Wrapper (Dùng tham số chuẩn của langchain-openai mới)
    langchain_llm = ChatOpenAI(
        model='deepseek-reasoner', 
        api_key=api_key,           # Sử dụng tham số api_key thay vì openai_api_key
        base_url=base_url,         # Sử dụng tham số base_url thay vì openai_api_base
        streaming=True
    )
    
    return client, dense_model, sparse_model, raw_llm, langchain_llm

COLLECTION_NAME = "hotpot_qa"