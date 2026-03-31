from langchain.tools import tool
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models
import streamlit as st

# Get resources
client, dense_model, sparse_model, raw_llm, _ = get_resources()

@tool
def rewrite_query_tool(query: str) -> str:
    """Use when the question depends on chat history or is ambiguous.
    This tool will create a standalone query."""
    # Take last 3 messages for context
    history = st.session_state.get("messages", [])[-3:]
    if not history: return query

    prompt = f"Chat History: {history}\nCurrent Question: {query}\nRewrite to a complete search query or reply 'ORIGINAL'."
    
    response = raw_llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    return query if "ORIGINAL" in result.upper() else result.strip()

@tool
def hybrid_search_tool(query_text: str) -> str:
    """Search for information in Qdrant Cloud using Hybrid Search (Dense + Sparse). 
    Returns the most relevant text snippets."""
    # Embed query for both dense and sparse
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(), 
        values=query_sparse_raw.values.tolist()
    )

    # Hybrid search on Qdrant
    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=15),
            models.Prefetch(query=query_sparse, using="sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5
    ).points
    
    if not points: return "No documents found for this query."
    
    # Format results with source titles
    return "\n\n".join([f"Source [{p.payload.get('title')}]: {p.payload.get('text')}" for p in points])

@tool
def hop2_expansion_tool(titles: list) -> str:
    """Use when you need to gather more details about specific entities found in the previous step.
    Pass in a list of titles."""
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="title", match=models.MatchAny(any=titles))]
        ),
        limit=5
    )[0]
    return "\n\n".join([f"Additional Info [{p.payload.get('title')}]: {p.payload.get('text')}" for p in results])

# Tools list for Agent
tools = [rewrite_query_tool, hybrid_search_tool, hop2_expansion_tool]