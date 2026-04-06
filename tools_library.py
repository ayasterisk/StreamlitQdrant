from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List
import json
from qdrant_client import models
from core_utils import get_resources, COLLECTION_NAME
import streamlit as st

# Lấy tài nguyên từ core_utils (đã được cache)
client, dense_model, sparse_model, langchain_llm = get_resources()

# --- Định nghĩa Pydantic Schemas cho đầu vào của Tool ---

class RewriteInput(BaseModel):
    query: str = Field(description="The user's original query that needs to be standalone.")

class HybridSearchInput(BaseModel):
    query_text: str = Field(description="The refined search query.")
    top_k: int = Field(default=5, description="Number of documents to retrieve. Use 10-15 for general/summary questions, 5 for specific ones.")

class Hop2Input(BaseModel):
    titles: List[str] = Field(description="List of document titles to fetch more context.")

# --- Các Tools ---

@tool("rewrite_query_tool", args_schema=RewriteInput)
def rewrite_query_tool(query: str) -> str:
    """
    OPTIONAL: Rewrite query if it's ambiguous or depends on chat history.
    If clear, returns 'ORIGINAL'.
    """
    history = st.session_state.get("messages", [])[-3:]
    if not history:
        return "ORIGINAL"

    prompt = f"Chat History: {history}\nCurrent Question: {query}\nRewrite this into a standalone search query. If already clear, return ONLY the word 'ORIGINAL'."
    
    response = langchain_llm.invoke(prompt)
    result = response.content.strip()
    
    return query if "ORIGINAL" in result.upper() else result

@tool("hybrid_search_tool", args_schema=HybridSearchInput)
def hybrid_search_tool(query_text: str, top_k: int = 5) -> str:
    """
    MANDATORY: Search the database for information. 
    Use top_k=10 or 15 for broad questions.
    """
    # 1. Embeddings
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    # 2. Dynamic Prefetch cho RRF
    prefetch_val = max(top_k * 3, 20)

    # 3. Query Qdrant
    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=prefetch_val),
            models.Prefetch(query=query_sparse, using="sparse", limit=prefetch_val),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    if not points:
        return json.dumps({"documents": []})

    results = [{
        "title": p.payload.get("title"),
        "text": p.payload.get("text"),
        "is_supporting": p.payload.get("is_supporting", False)
    } for p in points]

    return json.dumps({"source": "qdrant", "documents": results}, indent=2)

@tool("hop2_expansion_tool", args_schema=Hop2Input)
def hop2_expansion_tool(titles: List[str]) -> str:
    """
    FALLBACK: Expand search for specific titles to get more facts.
    """
    if not titles:
        return json.dumps({"documents": []})

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="title", match=models.MatchAny(any=titles))]
        ),
        limit=15
    )[0]

    docs = [{
        "title": p.payload.get("title"),
        "text": p.payload.get("text"),
        "is_supporting": p.payload.get("is_supporting", False)
    } for p in results]

    return json.dumps({"documents": docs}, indent=2)

tools = [rewrite_query_tool, hybrid_search_tool, hop2_expansion_tool]