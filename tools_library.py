from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, _ = get_resources()

# --- INPUT SCHEMAS ---

class SearchInput(BaseModel):
    query: str = Field(..., description="Standalone search query. Explicitly resolve all pronouns (he, she, it, that company) before calling.")
    prefetch_limit: int = Field(default=20, description="Number of candidates to fetch. Increase (40+) for complex/multi-entity queries.")
    final_limit: int = Field(default=5, description="Maximum number of document snippets to return.")

class ExpansionInput(BaseModel):
    follow_up_query: str = Field(..., description="Targeted question to find missing attributes of a known entity.")
    target_entities: List[str] = Field(..., description="Exact Titles/Names from previous search results to focus the search context.")

# --- TOOLS ---

@tool(args_schema=SearchInput)
def hybrid_search_tool(query: str, prefetch_limit: int = 20, final_limit: int = 5) -> str:
    """
    Description:
        The primary discovery tool. Performs a broad hybrid search (semantic + keyword) across the database. 
        Use this FIRST for any new question or to find initial leads/entities.
    
    Args:
        query (str): The search string. Must be specific and resolve pronouns from context.
        prefetch_limit (int): Depth of search. Default is 20, but use 40-60 for 'compare' or 'relationship' queries.
        final_limit (int): Number of snippets to return.
    
    Returns:
        str: A structured string starting with STATUS: SUCCESS/NOT_FOUND/ERROR and the document CONTENT (snippets).
    
    Raises:
        ValueError: If the query is empty or contains only whitespace.
        RuntimeError: If there's a connection failure with the Qdrant database or embedding models.
    """
    if not query.strip():
        return "STATUS: ERROR\nRAISES: ValueError - Search query cannot be empty."

    words = query.split()
    actual_pre = 40 if len(words) > 12 or "compare" in query.lower() else prefetch_limit

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
            limit=final_limit
        )

        if not response.points:
            return "STATUS: NOT_FOUND\nCONTENT: No relevant documents found."

        docs = []
        for p in response.points:
            payload = p.payload or {}
            title = payload.get('title', 'N/A')
            text = payload.get('text', '')

            snippet = (text[:400] + "...") if len(text) > 400 else text
            docs.append(f"Source: {title}\nSnippet: {snippet}")
        
        return f"STATUS: SUCCESS\nCONTENT: {' | '.join(docs)}"

    except Exception as e:
        return f"STATUS: ERROR\nRAISES: RuntimeError - {str(e)}"

@tool(args_schema=ExpansionInput)
def hop2_expansion_tool(follow_up_query: str, target_entities: List[str]) -> str:
    """
    Description:
        Precision 'Deep-Dive' tool. It bridges a known entity (found in a previous search) with missing specific details.
        Use this after you have identified a subject title from hybrid_search_tool but need more specific facts about it.
        Just use this tool only once to expand.
    
    Args:
        follow_up_query (str): The specific question about the identified subject (e.g., 'Who is the CEO?').
        target_entities (List[str]): List of 'Source' titles from previous SUCCESS results to anchor the search context.
    
    Returns:
        str: Expanded context snippets regarding the specific entities.
    
    Raises:
        AttributeError: If target_entities is not a valid list of strings.
    """
    if not isinstance(target_entities, list) or not target_entities:
        return "STATUS: ERROR\nRAISES: AttributeError - target_entities must be a non-empty list."

    context = " and ".join(target_entities)
    refined_query = f"{follow_up_query} specifically regarding {context}"
    
    return hybrid_search_tool.invoke({"query": refined_query, "prefetch_limit": 30})

tools = [hybrid_search_tool, hop2_expansion_tool]