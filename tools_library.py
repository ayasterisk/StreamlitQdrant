from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, raw_llm = get_resources()

class SearchInput(BaseModel):
    query: str = Field(..., description="The standalone search query. You MUST resolve all pronouns (he, it, that company) to specific names before calling this.")
    prefetch_limit: int = Field(default=20, description="Number of candidates to fetch from each index. Use 40-60 for complex, multi-entity queries.")
    final_limit: int = Field(default=5, description="The maximum number of top documents to return. Increase if the answer requires broad context.")

class ExpansionInput(BaseModel):
    follow_up_query: str = Field(..., description="A specific question to find missing details or attributes of a lead found in previous searches.")
    target_entities: List[str] = Field(..., description="A list of document titles or entity names identified in the first search to anchor the context.")

def is_complex_or_long(query: str) -> bool:
    keywords = ["and", "compare", "relationship", "difference", "between", "both"]
    return len(query.split()) > 12 or any(k in query.lower() for k in keywords)

def format_output(status: str, content: str, error_msg: Optional[str] = None) -> str:
    output = f"STATUS: {status}\n"
    if error_msg:
        output += f"RAISES: {error_msg}\n"
    output += f"CONTENT: {content}"
    return output

@tool(args_schema=SearchInput)
def hybrid_search_tool(query: str, prefetch_limit: int = 20, final_limit: int = 5) -> str:
    if not query.strip():
        return format_output("ERROR", "Query cannot be empty.", "ValueError")

    actual_pre = prefetch_limit
    actual_fin = final_limit
    if prefetch_limit == 20 and final_limit == 5 and is_complex_or_long(query):
        actual_pre, actual_fin = 40, 10

    try:
        query_dense = list(dense_model.embed([query]))[0]
        sparse_raw = list(sparse_model.embed([query]))[0]
        query_sparse = models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist()
        )

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=actual_pre),
                models.Prefetch(query=query_sparse, using="sparse", limit=actual_pre),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=actual_fin
        )

        if not response.points:
            return format_output("NOT_FOUND", "No documents matched the query.")

        docs = []
        for p in response.points:
            payload = p.payload or {}
            title = payload.get('title', 'N/A')
            text = payload.get('text', '')
            
            snippet = (text[:400] + "...") if len(text) > 400 else text
            docs.append(f"Title: {title}\nSnippet: {snippet}")
        
        return format_output("SUCCESS", "\n\n".join(docs))

    except Exception as e:
        return format_output("ERROR", "Search failed.", f"RuntimeError: {str(e)}")

@tool(args_schema=ExpansionInput)
def hop2_expansion_tool(follow_up_query: str, target_entities: List[str]) -> str:
    if not target_entities:
        return hybrid_search_tool.invoke({"query": follow_up_query})
    
    context = " and ".join(target_entities)
    refined_query = f"{follow_up_query} regarding {context}"
    return hybrid_search_tool.invoke({"query": refined_query, "prefetch_limit": 30})

tools = [hybrid_search_tool, hop2_expansion_tool]