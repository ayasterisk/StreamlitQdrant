import os
from langsmith import traceable, trace
import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI

st.set_page_config(page_title="HotpotQA RAG Agent", layout="wide")

# Environment variables for LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "hotpotqa-rag"

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_resources():
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    llm_client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    return client, dense_model, sparse_model, llm_client


client, dense_model, sparse_model, llm_client = init_resources()
COLLECTION_NAME = "hotpot_qa"

# Early stopping function
@traceable(name="Early Stop Decision")
def early_stop(results):
    if not results:
        return False, "No results"

    scores = [p.score for p in results if hasattr(p, "score")]

    if len(scores) < 2:
        return True, "Only 1 document → sufficient"

    top1, top2 = scores[0], scores[1]
    gap = top1 - top2

    titles = [p.payload.get("title", "") for p in results[:3]]
    unique_titles = len(set(titles))

    if top1 > 0.85:
        return True, "Top-1 score is very high"

    if gap > 0.2:
        return True, "Large score gap"

    if unique_titles == 1 and top1 > 0.75:
        return True, "Single document is strong enough"

    return False, "Need multi-hop"

# Query rewriting function
@traceable(name="Rewrite Query with History")
def rewrite_query_with_history(query, history):
    if len(history) < 2:
        return query

    history_text = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in history[-4:]
    ])

    prompt = f"""
Rewrite the question to be self-contained.

Chat history:
{history_text}

Question:
{query}

Rewritten question:
"""

    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return query

# Main retrieval function
@traceable(name="Hybrid Retrieval + Multi-hop")
def advanced_retrieval(query_text, top_k=5):
    
    # Embed query
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]

    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    hop1_points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    hop1_points = sorted(
        hop1_points,
        key=lambda x: getattr(x, "score", 0),
        reverse=True
    )

    if len(hop1_points) <= 1:
        trace.log("Early Stop", "Only 1 document retrieved")
        return hop1_points, "Early Stop (1 document)"

    should_stop, reason = early_stop(hop1_points)

    if should_stop:
        trace.log("Strategy", f"Early Stop: {reason}")
        return hop1_points, f"Early Stop ({reason})"

    # HOP-2
    final_evidence = list(hop1_points)
    seen_ids = {p.id for p in final_evidence}

    bridge_titles = {
        p.payload.get("title")
        for p in hop1_points
        if p.payload.get("is_supporting")
    }

    if bridge_titles:
        hop2_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchAny(any=list(bridge_titles))
                    ),
                    models.FieldCondition(
                        key="is_supporting",
                        match=models.MatchValue(value=True)
                    )
                ]
            ),
            limit=10
        )[0]

        for p in hop2_points:
            if p.id not in seen_ids:
                final_evidence.append(p)
                seen_ids.add(p.id)

    trace.log({
        "query": query_text,
        "num_docs": len(final_evidence),
        "titles": [p.payload.get("title") for p in final_evidence],
        "scores": [getattr(p, "score", None) for p in final_evidence],
        "is_supporting": [p.payload.get("is_supporting") for p in final_evidence]
    })
    
    return final_evidence, "Full Multi-hop"

# Context building function
@traceable(name="Build Context")
def build_context(evidence):
    return "\n\n".join([
        f"[{i+1}] {p.payload.get('title')}\n{p.payload.get('text')}"
        for i, p in enumerate(evidence)
    ])

# LLM reasoning function
@traceable(name="LLM Reasoning")
def generate_answer(llm_client, history_for_llm):
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=history_for_llm,
        temperature=0.3
    )
    return response.choices[0].message.content

# Full RAG pipeline function
@traceable(name="Full RAG Pipeline")
def rag_pipeline(query, history):
    rewritten_query = rewrite_query_with_history(query, history)
    evidence, strategy = advanced_retrieval(rewritten_query)
    context = build_context(evidence)

    return rewritten_query, evidence, strategy, context


# Streamlit UI
st.title("HotpotQA RAG Agent")

# History display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Enter your question...")

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # Rewrite query
    rewritten_query = rewrite_query_with_history(
        query, st.session_state.messages
    )

    with st.status("Retrieving documents...", expanded=True):
        evidence, strategy = advanced_retrieval(rewritten_query)

        st.write(f"Strategy: **{strategy}**")

        context = "\n\n".join([
            f"[{i+1}] {p.payload.get('title')}\n{p.payload.get('text')}"
            for i, p in enumerate(evidence)
        ])

    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            prompt = f"""You are a multi-hop question answering system.

RULES:
1. CITATION: Always cite sources using [1], [2], etc.
2. COMPARISON: If it's a comparison question, analyze each entity before concluding.
3. HONESTY: If the answer is not in the documents, say "I do not have enough information to answer this question."
4. USE EVIDENCE: Base your reasoning on the provided documents, not only general knowledge.
5. FINAL ANSWER: Provide a clear and concise final answer.

DOCUMENTS:
{context}

QUESTION:
{query}

ANSWER:
"""

            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:]
            ]

            history_for_llm.append({
                "role": "user",
                "content": prompt
            })

            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=history_for_llm,
                    temperature=0.3
                )

                answer = response.choices[0].message.content
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"API Error: {e}")

    with st.expander("Debug Metadata"):
        st.json([
            {
                "title": p.payload.get("title"),
                "text": p.payload.get("text"),
                "score": getattr(p, "score", None),
                "is_supporting": p.payload.get("is_supporting")
            }
            for p in evidence
        ])

# Reset
if st.button("Clear chat history"):
    st.session_state.messages = []
    st.rerun()