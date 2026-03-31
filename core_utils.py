import os
import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI
from langchain_openai import ChatOpenAI

@st.cache_resource
def get_resources():
    # 1. Kích hoạt LangSmith Tracing (Dùng string để an toàn tuyệt đối)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if st.secrets.get("LANGCHAIN_TRACING_V2") else "false"
    os.environ["LANGCHAIN_API_KEY"] = str(st.secrets.get("LANGCHAIN_API_KEY", ""))
    os.environ["LANGCHAIN_PROJECT"] = str(st.secrets.get("LANGCHAIN_PROJECT", "Multi-hop-RAG"))

    # 2. Kết nối Qdrant Cloud
    client = QdrantClient(
        url=str(st.secrets["QDRANT_URL"]),
        api_key=str(st.secrets["QDRANT_API_KEY"])
    )
    
    # 3. Khởi tạo Embedding Models
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    # Làm sạch thông tin DeepSeek
    ds_api_key = str(st.secrets["DEEPSEEK_API_KEY"]).strip()
    ds_base_url = "https://api.deepseek.com"

    # 4. Khởi tạo DeepSeek Client chuẩn (Bỏ qua các tham số mặc định của hệ thống)
    # Lỗi TypeError thường do tham số base_url hoặc api_key bị nhận diện sai kiểu
    raw_llm = OpenAI(
        api_key=ds_api_key,
        base_url=ds_base_url
    )
    
    # 5. LangChain LLM Wrapper 
    langchain_llm = ChatOpenAI(
        model='deepseek-reasoner', 
        openai_api_key=ds_api_key,
        openai_api_base=ds_base_url,
        max_retries=3,
        streaming=True
    )
    
    return client, dense_model, sparse_model, raw_llm, langchain_llm

COLLECTION_NAME = "hotpot_qa"