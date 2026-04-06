from langchain.tools import tool
from pydantic import BaseModel, Field
import json
from qdrant_client import models
from core_utils import get_resources, COLLECTION_NAME

client, dense_model, sparse_model, _ = get_resources()

class HybridSearchSchema(BaseModel):
    query: str = Field(description="Search query for the database.")
    top_k: int = Field(default=5, description="Number of docs to fetch. Use 10-15 for summaries/broad questions.")

@tool("hybrid_search_tool", args_schema=HybridSearchSchema)
def hybrid_search_tool(query: str, top_k: int = 5) -> str:
    """MANDATORY: Search the Qdrant database."""
    
    # Vectorize
    dense_vec = list(dense_model.embed([query]))[0].tolist()
    sparse_raw = list(sparse_model.embed([query]))[0]
    sparse_vec = models.SparseVector(
        indices=sparse_raw.indices.tolist(), 
        values=sparse_raw.values.tolist()
    )

    # Dynamic Prefetch dựa trên top_k
    prefetch_limit = max(top_k * 2, 20)

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            models.Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    docs = [{"title": p.payload.get("title"), "text": p.payload.get("text")} for p in points]
    return json.dumps(docs, indent=2)

tools = [hybrid_search_tool] # Thêm các tool khác vào đây