from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, _ = get_resources()

class SearchInput(BaseModel):
    query: str = Field(..., description="Standalone search query. Resolve pronouns first.")
    prefetch_limit: int = Field(default=20)
    final_limit: int = Field(default=5)

class ExpansionInput(BaseModel):
    follow_up_query: str = Field(..., description="Targeted query for missing details.")
    target_entities: List[str] = Field(..., description="Entities to anchor the search.")

def is_complex_or_long(query: str) -> bool:
    keywords = ["and", "compare", "relationship", "difference", "between", "both"]
    return len(query.split()) > 12 or any(k in query.lower() for k in keywords)

def format_output(status: str, content: str, error_msg: Optional[str] = None) -> str:
    output = f"STATUS: {status}\n"
    if error_msg: output += f"RAISES: {error_msg}\n"
    output += f"CONTENT: {content}"
    return output

@tool(args_schema=SearchInput)
def hybrid_search_tool(query: str, prefetch_limit: int = 20, final_limit: int = 5) -> str:
    """Perform hybrid search. Primary tool for fact discovery."""
    if not query.strip(): return format_output("ERROR", "Empty query.", "ValueError")

    actual_pre = 40 if is_complex_or_long(query) else prefetch_limit
    
    try:
        query_dense = list(dense_model.embed([query]))[0]
        sparse_raw = list(sparse_model.embed([query]))[0]
        query_sparse = models.SparseVector(indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist())

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=actual_pre),
                models.Prefetch(query=query_sparse, using="sparse", limit=actual_pre),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=final_limit
        )

        if not response.points: return format_output("NOT_FOUND", "No results.")

        docs = []
        for p in response.points:
            payload = p.payload or {}
            text = payload.get('text', '')
            snippet = (text[:400] + "...") if len(text) > 400 else text
            docs.append(f"Source: {payload.get('title')}\nSnippet: {snippet}")
        
        return format_output("SUCCESS", "\n\n".join(docs))
    except Exception as e:
        return format_output("ERROR", "Search failed.", str(e))

@tool(args_schema=ExpansionInput)
def hop2_expansion_tool(follow_up_query: str, target_entities: List[str]) -> str:
    """Bridge gaps between entities found in previous steps. Just use this tool only once to expand."""
    context = " and ".join(target_entities)
    return hybrid_search_tool.invoke({"query": f"{follow_up_query} regarding {context}", "prefetch_limit": 30})

tools = [hybrid_search_tool, hop2_expansion_tool]