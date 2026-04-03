from langchain.tools import tool
from pydantic import BaseModel, Field
import json
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, _ = get_resources()

class HybridSearchInput(BaseModel):
    query_text: str = Field(
        description="The standalone search query. Must be descriptive and resolve any pronouns from history."
    )
    limit: int = Field(
        default=10, 
        description="Number of top results to return. Use 15-20 for broad topics, 5-10 for specific facts."
    )

class Hop2Input(BaseModel):
    titles: list[str] = Field(
        description="A list of exact document titles to expand and get full content."
    )

@tool(args_schema=HybridSearchInput)
def hybrid_search_tool(query_text: str, limit: int = 10) -> str:
    """
    Search the knowledge base using hybrid retrieval (Dense + Sparse).
    Use this to find initial evidence, snippets, and document titles.
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
            # Prefetch cố định ở 50 để tối ưu hóa chất lượng RRF
            models.Prefetch(query=query_dense, using="dense", limit=50),
            models.Prefetch(query=query_sparse, using="sparse", limit=50),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    ).points

    results = []
    for p in points:
        results.append({
            "title": p.payload.get("title"),
            "text": p.payload.get("text"),
            "is_supporting": p.payload.get("is_supporting")
        })
    return json.dumps(results, ensure_ascii=False)


@tool(args_schema=Hop2Input)
def hop2_expansion_tool(titles: list[str]) -> str:
    """
    Fetch full detailed text for a list of specific titles. 
    Use this only if hybrid_search_tool's snippet is insufficient.
    """
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