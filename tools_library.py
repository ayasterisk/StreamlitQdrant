from langchain.tools import tool
from pydantic import BaseModel, Field
import json
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models
import streamlit as st

client, dense_model, sparse_model, raw_llm = get_resources()

# =========================
# 🔹 SCHEMA
# =========================
class QueryInput(BaseModel):
    query: str = Field(..., description="User question")

class TitlesInput(BaseModel):
    titles: list[str] = Field(..., description="List of document titles")

# =========================
# 🔹 HELPER
# =========================
def is_complex_query(query: str):
    keywords = ["and", "or", "compare", "relationship", "difference", "both"]
    return any(k in query.lower() for k in keywords)

def is_general_query(query: str):
    return len(query.split()) > 12

# =========================
# 🔹 TOOL 1: REWRITE
# =========================
@tool(args_schema=QueryInput)
def rewrite_query_tool(query: str) -> str:
    history = st.session_state.get("messages", [])[-3:]

    if not history or len(query.split()) > 8:
        return "ORIGINAL"

    prompt = f"""
    Chat History: {history}
    Question: {query}

    Rewrite into standalone query.
    If clear → return "ORIGINAL"
    """

    response = raw_llm.invoke(prompt)
    result = response.content.strip()

    if "ORIGINAL" in result.upper():
        return "ORIGINAL"
    return result

# =========================
# 🔹 TOOL 2: HYBRID SEARCH (DYNAMIC)
# =========================
@tool(args_schema=QueryInput)
def hybrid_search_tool(query: str) -> str:

    if is_general_query(query):
        prefetch_limit = 50
        final_limit = 12
    elif is_complex_query(query):
        prefetch_limit = 30
        final_limit = 8
    else:
        prefetch_limit = 15
        final_limit = 5

    query_dense = list(dense_model.embed([query]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query]))[0]

    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=prefetch_limit),
            models.Prefetch(query=query_sparse, using="sparse", limit=prefetch_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=final_limit
    ).points

    if not points:
        return json.dumps({"documents": []}, indent=2)

    results = []
    for p in points:
        results.append({
            "title": p.payload.get("title"),
            "text": p.payload.get("text"),
            "is_supporting": p.payload.get("is_supporting", False)
        })

    return json.dumps({"documents": results}, indent=2)

# =========================
# 🔹 TOOL 3: HOP2 (IMPROVED)
# =========================
@tool(args_schema=TitlesInput)
def hop2_expansion_tool(titles: list[str]) -> str:
    if not titles:
        return json.dumps({"documents": []})
    query = " ".join(titles)

    return hybrid_search_tool.invoke({"query": query})

# =========================
tools = [
    rewrite_query_tool,
    hybrid_search_tool,
    hop2_expansion_tool
]