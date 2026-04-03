from langchain.tools import tool
import json
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, _ = get_resources()

@tool
def hybrid_search_tool(query_text: str) -> str:
    """Search the knowledge base for relevant snippets and titles. 
    Returns: title, text, and is_supporting flag."""
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=25),
            models.Prefetch(query=query_sparse, using="sparse", limit=25),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=8
    ).points

    results = []
    for p in points:
        results.append({
            "title": p.payload.get("title"),
            "text": p.payload.get("text"),
            "is_supporting": p.payload.get("is_supporting")
        })
    return json.dumps(results, ensure_ascii=False)

@tool
def hop2_expansion_tool(titles: list[str]) -> str:
    """Retrieve full detailed text for specific document titles."""
    if not titles: return "[]"
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="title", match=models.MatchAny(any=titles))]
        ),
        limit=len(titles)
    )
    docs = [{"title": p.payload.get("title"), "text": p.payload.get("text")} for p in results]
    return json.dumps(docs, ensure_ascii=False)

tools = [hybrid_search_tool, hop2_expansion_tool]