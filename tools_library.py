from langchain.tools import tool
import json
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models
import streamlit as st

# Get resources
client, dense_model, sparse_model, raw_llm, _ = get_resources()


@tool
def hybrid_search_tool(query_text: str) -> str:
    """
    MANDATORY retrieval tool.
    """

    history = st.session_state.get("messages", [])[-3:]

    context = " ".join([
        m["content"] for m in history
        if m["role"] in ["user", "assistant"]
    ])
    
    final_query = f"{context} {query_text}".strip()

    query_dense = list(dense_model.embed([final_query]))[0].tolist()

    query_sparse_raw = list(sparse_model.embed([final_query]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=15),
            models.Prefetch(query=query_sparse, using="sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5
    ).points

    if not points:
        return json.dumps({"source": "qdrant", "documents": []}, indent=2)

    results = []
    for p in points:
        results.append({
            "title": p.payload.get("title"),
            "text": p.payload.get("text"),
            "is_supporting": p.payload.get("is_supporting", False)
        })

    return json.dumps({
        "source": "qdrant",
        "documents": results
    }, indent=2)


@tool
def hop2_expansion_tool(titles: list) -> str:
    """
    Fallback multi-hop tool.
    """

    if not titles:
        return json.dumps({"documents": []})

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="title",
                    match=models.MatchAny(any=titles)
                )
            ]
        ),
        limit=10
    )[0]

    if not results:
        return json.dumps({"documents": []})

    supporting_docs = []
    other_docs = []

    for p in results:
        doc = {
            "title": p.payload.get("title"),
            "text": p.payload.get("text"),
            "is_supporting": p.payload.get("is_supporting", False)
        }

        if doc["is_supporting"]:
            supporting_docs.append(doc)
        else:
            other_docs.append(doc)

    final_docs = supporting_docs + other_docs

    return json.dumps({
        "documents": final_docs
    }, indent=2)


tools = [
    hybrid_search_tool,
    hop2_expansion_tool
]