from langchain.tools import tool
from pydantic import BaseModel, Field
import json
from typing import List
from core_utils import get_resources, COLLECTION_NAME
from qdrant_client import models

client, dense_model, sparse_model, raw_llm = get_resources()

class QueryInput(BaseModel):
    query: str = Field(..., description="User question")

class TitlesInput(BaseModel):
    titles: List[str] = Field(..., description="List of document titles")

def is_complex_query(query: str):
    keywords = ["and", "or", "compare", "relationship", "difference", "both"]
    return any(k in query.lower() for k in keywords)

def is_general_query(query: str):
    return len(query.split()) > 12

@tool(args_schema=QueryInput)
def rewrite_query_tool(query: str) -> str:
    """
    Use rewrite_query_tool only when absolutely necessary, for example, when a clear entity cannot be found in the query.
    Return "ORIGINAL" if no rewrite needed.
    """

    if len(query.split()) > 8 and not is_complex_query(query):
        return "ORIGINAL"

    prompt = f"""
Rewrite the following question into a clear standalone query.

Question: {query}

Rules:
- If already clear → return EXACTLY "ORIGINAL"
- Do NOT add explanation
"""

    try:
        response = raw_llm.invoke(prompt)
        result = response.content.strip()

        if "ORIGINAL" in result.upper():
            return "ORIGINAL"

        return result

    except Exception:
        return "ORIGINAL"

@tool(args_schema=QueryInput)
def hybrid_search_tool(query: str) -> str:
    """
    Hybrid search using RRF (dense + sparse).
    MUST be used before answering.
    """

    if is_general_query(query):
        prefetch_limit = 40
        final_limit = 10
    elif is_complex_query(query):
        prefetch_limit = 25
        final_limit = 8
    else:
        prefetch_limit = 12
        final_limit = 5

    query_dense = list(dense_model.embed([query]))[0]

    sparse_raw = list(sparse_model.embed([query]))[0]
    query_sparse = models.SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist()
    )

    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=prefetch_limit),
                models.Prefetch(query=query_sparse, using="sparse", limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=final_limit
        )

        points = response.points

    except Exception:
        return json.dumps({"documents": []})

    if not points:
        return json.dumps({"documents": []})

    seen = set()
    results = []

    for p in points:
        payload = p.payload or {}

        text = payload.get("text", "")
        title = payload.get("title", "")

        if not text or text in seen:
            continue

        seen.add(text)

        results.append({
            "title": title,
            "text": text[:500],
            "is_supporting": payload.get("is_supporting", False)
        })

    return json.dumps({"documents": results}, indent=2)

@tool(args_schema=TitlesInput)
def hop2_expansion_tool(titles: List[str]) -> str:
    """
    Second-hop retrieval using titles.
    """
    titles = [t for t in titles if t and isinstance(t, str)]

    if not titles:
        return json.dumps({"documents": []})

    query = " ; ".join(titles)

    return hybrid_search_tool.invoke({"query": query})

tools = [
    rewrite_query_tool,
    hybrid_search_tool,
    hop2_expansion_tool
]