import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI

# --- UI CONFIGURATION ---
st.set_page_config(page_title="DeepSeek-R1 Multi-hop Agent", layout="wide", page_icon="🧠")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RESOURCE INITIALIZATION ---
@st.cache_resource
def init_resources():
    # Initialize Qdrant Client
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    
    # Initialize Embedding Models
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    # Initialize DeepSeek Client
    llm_client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )
    return client, dense_model, sparse_model, llm_client

client, dense_model, sparse_model, llm_client = init_resources()
COLLECTION_NAME = "hotpot_qa"

# --- DEEPSEEK-REASONER HELPER FUNCTION ---
def call_reasoner(messages):
    """Calls DeepSeek-R1 and returns (reasoning_content, final_content)"""
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        reasoning = getattr(response.choices[0].message, 'reasoning_content', "No reasoning provided.")
        content = response.choices[0].message.content
        return reasoning, content
    except Exception as e:
        st.error(f"API Error: {e}")
        return "", "I encountered an error connecting to the AI service."

# --- LOGIC 1: SMART QUERY REWRITE ---
def smart_rewrite(query, history):
    """Rewrites the query only if it depends on conversation history"""
    if not history:
        return query, False

    prompt = f"""
    Analyze the chat history and the current question.
    History: {history[-3:]}
    Question: "{query}"

    Task:
    - If the question is standalone (e.g., "Who is Steve Jobs?"), respond exactly with: ORIGINAL
    - If the question refers to previous topics (e.g., "When was he born?"), rewrite it into a complete search query.

    Format: Respond only with the rewritten text or the word ORIGINAL.
    """
    _, result = call_reasoner([{"role": "user", "content": prompt}])
    
    if "ORIGINAL" in result.upper() and len(result) < 15:
        return query, False
    return result.strip(), True

# --- LOGIC 2: ADVANCED RETRIEVAL (HYBRID + MULTI-HOP) ---
def advanced_retrieval(query_text):
    """Performs Hybrid Search and decides if a second hop is needed using R1 reasoning"""
    
    # Generate Embeddings
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(), 
        values=query_sparse_raw.values.tolist()
    )

    # Step 1: Hop-1 Search
    hop1_points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=15),
            models.Prefetch(query=query_sparse, using="sparse", limit=15),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5
    ).points

    if not hop1_points:
        return [], "No results found", "No relevant documents were found in the database."

    # Step 2: Early Stop Reasoning via R1
    context_preview = "\n".join([f"- {p.payload.get('title')}: {p.payload.get('text')[:200]}..." for p in hop1_points[:2]])
    stop_prompt = f"""
    Question: {query_text}
    Top Retrieved Docs:
    {context_preview}

    Does the information above fully answer the question, or do you need to look up more bridge entities?
    Reply 'STOP' if sufficient, otherwise reply 'CONTINUE' with a brief reason.
    """
    reasoning, decision = call_reasoner([{"role": "user", "content": stop_prompt}])
    
    if "STOP" in decision.upper():
        return hop1_points, "Early Stop (Sufficient Information)", reasoning

    # Step 3: Hop-2 Retrieval (Fetch linked documents by Title)
    bridge_titles = [p.payload.get("title") for p in hop1_points[:3]]
    hop2_points = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="title", match=models.MatchAny(any=bridge_titles))]
        ),
        limit=5
    )[0]

    final_evidence = list(hop1_points)
    seen_ids = {p.id for p in final_evidence}
    for p in hop2_points:
        if p.id not in seen_ids:
            final_evidence.append(p)
            seen_ids.add(p.id)

    return final_evidence, "Full Multi-hop Retrieval", reasoning

# --- STREAMLIT UI ---
st.title("Multi-hop Agent")
st.caption("RAG System")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
query = st.chat_input("Ask a complex question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        # 1. QUERY ANALYSIS & REWRITING
        with st.status("Analyzing context...", expanded=False) as status:
            rewritten_q, was_rewritten = smart_rewrite(query, st.session_state.messages[:-1])
            if was_rewritten:
                st.write(f"Optimized Query: **{rewritten_q}**")
            else:
                st.write("Query is standalone. No rewrite needed.")
            status.update(label="Query Analysis Complete", state="complete")

        # 2. KNOWLEDGE RETRIEVAL
        with st.status("Retrieving documents...", expanded=True):
            evidence, strategy, stop_logic = advanced_retrieval(rewritten_q)
            st.write(f"Retrieval Strategy: **{strategy}**")
            with st.expander("AI Thinking (Retriever Level)"):
                st.markdown(stop_logic)
        
        # 3. FINAL REASONING & ANSWERING
        context = "\n\n".join([f"[{i+1}] {p.payload.get('title')}: {p.payload.get('text')}" for i, p in enumerate(evidence)])
        final_prompt = f"""Use the following documents to answer the user's question accurately. 
        Always cite sources using [1], [2], etc.
        
        DOCUMENTS:
        {context}
        
        QUESTION:
        {query}
        """

        with st.spinner("DeepSeek is thinking..."):
            full_reasoning, final_answer = call_reasoner([
                {"role": "system", "content": "You are a logical RAG assistant. Use step-by-step thinking to verify facts against the provided documents."},
                {"role": "user", "content": final_prompt}
            ])

            # Display R1 Thinking Process
            with st.expander("Thinking Process", expanded=True):
                st.markdown(full_reasoning)

            # Display Final Answer
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # Debug Metadata Expanders
    with st.expander("Source Metadata"):
        st.json([{
            "title": p.payload.get("title"),
            "text": p.payload.get("text")[:200],
            "score": getattr(p, "score", None)} for p in evidence])

# Sidebar Controls
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()