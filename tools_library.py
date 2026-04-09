from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, raw_llm = get_resources()


class SearchInput(BaseModel):
    query: str = Field(
        ..., 
        description="The standalone search query. You MUST resolve all pronouns (he, it, that company) to specific names before calling this."
    )
    prefetch_limit: int = Field(
        default=20, 
        description="Number of candidates to fetch from each index. Use 40-60 for complex, multi-entity queries."
    )
    final_limit: int = Field(
        default=5, 
        description="The maximum number of top documents to return. Increase if the answer requires broad context."
    )

class ExpansionInput(BaseModel):
    follow_up_query: str = Field(
        ..., 
        description="A specific question to find missing details or attributes of a lead found in previous searches."
    )
    target_entities: List[str] = Field(
        ..., 
        description="A list of document titles or entity names identified in the first search to anchor the context."
    )


def is_complex_or_long(query: str) -> bool:
    """Helper to detect if a query needs more search depth."""
    keywords = ["and", "compare", "relationship", "difference", "between", "both"]
    return len(query.split()) > 12 or any(k in query.lower() for k in keywords)

def format_output(status: str, content: str, error_msg: Optional[str] = None) -> str:
    """Standardizes tool output for the agent's reasoning engine."""
    output = f"STATUS: {status}\n"
    if error_msg:
        output += f"RAISES: {error_msg}\n"
    output += f"CONTENT: {content}"
    return output


@tool(args_schema=SearchInput)
def hybrid_search_tool(query: str, prefetch_limit: int = 20, final_limit: int = 5) -> str:
    """
    Description:
        The primary fact-finding tool. Performs a hybrid search (Dense + Sparse) in the knowledge base.
        Use this for initial discovery or when you need general information about a topic.
    
    Args:
        query (str): The search query. Must be explicit (no pronouns).
        prefetch_limit (int): Candidates per index before fusion. Higher values improve recall for hard queries.
        final_limit (int): Number of documents returned to you.
    
    Returns:
        str: A structured string with STATUS (SUCCESS/NOT_FOUND/ERROR) and the document CONTENT.
    
    Raises:
        ValueError: If the query is empty.
        RuntimeError: If the search engine fails.
    """
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
            docs.append(f"Title: {payload.get('title')}\nText: {payload.get('text')}")
        
        return format_output("SUCCESS", "\n\n".join(docs))

    except Exception as e:
        return format_output("ERROR", "Search failed.", f"RuntimeError: {str(e)}")

@tool(args_schema=ExpansionInput)
def hop2_expansion_tool(follow_up_query: str, target_entities: List[str]) -> str:
    """
    Description:
        A precision tool for 'multi-hop' reasoning. Use this to bridge a lead found in an 
        initial search to a specific missing attribute (e.g., finding the spouse of an actor found in step 1).
    
    Args:
        follow_up_query (str): The specific question to resolve the gap.
        target_entities (List[str]): List of titles/entities from previous results to anchor the search.
    
    Returns:
        str: Targeted document content.
    """
    if not target_entities:
        return hybrid_search_tool.invoke({"query": follow_up_query})
    
    context = " and ".join(target_entities)
    refined_query = f"{follow_up_query} regarding {context}"
    
    return hybrid_search_tool.invoke({"query": refined_query, "prefetch_limit": 30})

tools = [hybrid_search_tool, hop2_expansion_tool]