from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
import json
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models
import streamlit as st

# SỬA LỖI TẠI ĐÂY: Chỉ unpack 4 biến
client, dense_model, sparse_model, llm = get_resources()

# --- Định nghĩa Pydantic Schemas cho Tool Inputs ---

class RewriteInput(BaseModel):
    query: str = Field(description="The user's original question that might need rewriting for clarity.")

class HybridSearchInput(BaseModel):
    query_text: str = Field(description="The search query for the database.")
    top_k: int = Field(default=5, description="Number of documents to retrieve. Use 10-15 for broad/general questions, 5 for specific ones.")

class Hop2Input(BaseModel):
    titles: List[str] = Field(description="List of document titles to expand and get more context.")

# --- Các Tools với Schema ---

@tool("rewrite_query_tool", args_schema=RewriteInput)
def rewrite_query_tool(query: str) -> str:
    """
    OPTIONAL: Use when the question is vague or depends on history.
    Returns 'ORIGINAL' if no rewrite is needed.
    """
    # Lấy history từ streamlit session state nếu cần
    history = st.session_state.get("messages", [])[-3:]
    if not history or len(query.split()) > 10:
        return "ORIGINAL"

    prompt = f"Chat History: {history}\nQuestion: {query}\nRewrite as a standalone query or return 'ORIGINAL'."
    
    # Giả định llm có phương thức invoke hoặc chat
    try:
        # Nếu llm là LangChain ChatOpenAI
        response = llm.invoke(prompt)
        result = response.content.strip()
    except:
        # Fallback nếu là raw OpenAI client
        response = llm.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
        result = response.choices[0].message.content.strip()

    return query if "ORIGINAL" in result.upper() else result

@tool("hybrid_search_tool", args_schema=HybridSearchInput)
def hybrid_search_tool(query_text: str, top_k: int = 5) -> str:
    """
    MANDATORY search tool. 
    Set top_k=15 for general/summary questions, top_k=5 for specific facts.
    """
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    # Thiết lập prefetch động: Luôn lấy nhiều hơn top_k để RRF xếp hạng tốt hơn
    prefetch_limit = max(top_k * 3, 20)

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=prefetch_limit),
            models.Prefetch(query=query_sparse, using="sparse", limit=prefetch_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    results = [{
        "title": p.payload.get("title"),
        "text": p.payload.get("text"),
        "is_supporting": p.payload.get("is_supporting", False)
    } for p in points]

    return json.dumps({"source": "qdrant", "documents": results}, indent=2)

@tool("hop2_expansion_tool", args_schema=Hop2Input)
def hop2_expansion_tool(titles: List[str]) -> str:
    """Expand search by getting all chunks related to specific titles."""
    if not titles: return json.dumps({"documents": []})

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="title", match=models.MatchAny(any=titles))]),
        limit=15
    )[0]

    docs = [{"title": p.payload.get("title"), "text": p.payload.get("text")} for p in results]
    return json.dumps({"documents": docs}, indent=2)

tools = [rewrite_query_tool, hybrid_search_tool, hop2_expansion_tool]