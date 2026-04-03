from langchain.tools import tool
import json
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, langchain_llm = get_resources()

@tool
def hybrid_search_tool(query_text: str) -> str:
    """
    Search information in the database. 
    Input MUST be a standalone search query with full context (names, entities).
    """
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
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

    results = []
    for p in points:
        results.append({
            "title": p.payload.get("title"),
            "text": p.payload.get("text")
        })

    return json.dumps({"documents": results}, indent=2)

@tool
def hop2_expansion_tool(titles: list) -> str:
    """Useful to get more details about specific article titles.
    Can skip if has enough information from hybrid_search_tool to answer the question."""
    if not titles: return "[]"
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="title", match=models.MatchAny(any=titles))]
        ),
        limit=5
    )[0]
    
    docs = [{"title": p.payload.get("title"), "text": p.payload.get("text")} for p in results]
    return json.dumps(docs, indent=2)

tools = [hybrid_search_tool, hop2_expansion_tool]