from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from core_utils import get_resources
from tools_library import tools
from trim import trim_messages
import streamlit as st

_, _, _, _, langchain_llm = get_resources()

@st.cache_resource
def get_shared_memory():
    return InMemorySaver()

def get_agent_executor():
    memory = get_shared_memory()

    system_prompt = """You are a retrieval-only QA assistant. Answer ONLY based on tool outputs.

    THINKING PROCESS:
    1. REWRITE: If the user question uses pronouns (e.g., "they", "it"), rewrite it into a standalone query using chat history.
    2. SEARCH: Call 'hybrid_search_tool' with the query.
    3. EVALUATE & ACT:
       - If the retrieved text ALREADY contains the answer -> Provide the answer immediately and STOP.
       - If the text is INSUFFICIENT to answer -> Identify the relevant 'titles' from the search results to expand. Use titles where 'is_supporting' is true as primary leads for expansion.
       - Call 'hop2_expansion_tool' ONLY if you need more details for those specific titles to finalize your answer.

    RULES:
    - ALWAYS cite sources as [title].
    - NEVER answer using your own knowledge.
    - NEVER call the same tool with the same arguments twice.
    - If no information is found after both steps, say "I don't know based on the database."

    Answer in English. Concise and direct."""

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[trim_messages],
        checkpointer=memory,
    )
    return agent