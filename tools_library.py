from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List
import json
from qdrant_client import models
from core_utils import get_resources, COLLECTION_NAME
import streamlit as st

# Lấy resources
client, dense_model, sparse_model, langchain_llm = get_resources()

# --- Schemas ---
class HybridSearchInput(BaseModel):
    query_text: str = Field(description="Search query.")
    top_k: int = Field(default=5, description="Number of docs (5 specific, 15 general).")

class RewriteInput(BaseModel):
    query: str = Field(description="Original query.")

class Hop2Input(BaseModel):
    titles: List[str] = Field(description="Titles to expand.")

# --- Tools ---
@tool("rewrite_query_tool", args_schema=RewriteInput)
def rewrite_query_tool(query: str) -> str:
    """Rewrite query based on chat history. Returns 'ORIGINAL' if clear."""
    history = st.session_state.get("messages", [])[-3:]
    if not history: return "ORIGINAL"
    
    prompt = f"History: {history}\nQuery: {query}\nRewrite to standalone or reply 'ORIGINAL'."
    res = langchain_llm.invoke(prompt)
    return query if "ORIGINAL" in res.content.upper() else res.content.strip()

@tool("hybrid_search_tool", args_schema=HybridSearchInput)
def hybrid_search_tool(query_text: str, top_k: int = 5) -> str:
    """MANDATORY: Search database. top_k is dynamic."""
    # Vectorize
    dense_vec = list(dense_model.embed([query_text]))[0].tolist()
    sparse_raw = list(sparse_model.embed([query_text]))[0]
    sparse_vec = models.SparseVector(
        indices=sparse_raw.indices.tolist(), 
        values=sparse_raw.values.tolist()
    )

    # RRF Search
    prefetch_limit = max(top_k * 3, 20)
    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            models.Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    results = [{"title": p.payload.get("title"), "text": p.payload.get("text")} for p in points]
    return json.dumps({"documents": results}, indent=2)

@tool("hop2_expansion_tool", args_schema=Hop2Input)
def hop2_expansion_tool(titles: List[str]) -> str:
    """Fallback to get more context for specific titles."""
    if not titles: return "[]"
    hits = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="title", match=models.MatchAny(any=titles))]),
        limit=10
    )[0]
    results = [{"title": h.payload.get("title"), "text": h.payload.get("text")} for h in hits]
    return json.dumps(results, indent=2)

tools = [rewrite_query_tool, hybrid_search_tool, hop2_expansion_tool]