from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from core_utils import get_resources
from tools_library import tools
from trim import trim_messages
import streamlit as st

_, _, _, langchain_llm = get_resources()

@st.cache_resource
def get_shared_memory():
    return InMemorySaver()

def get_agent_executor():
    memory = get_shared_memory()

    system_prompt = """You are a precise RAG assistant.
    
    CRITICAL INSTRUCTIONS FOR FOLLOW-UP QUESTIONS:
    1. INTENT SHIFT: When a user asks a follow-up question (e.g., "Who are they?", "Tell me more"), your goal is to provide NEW information. Do NOT repeat the logic or answers from previous turns (like nationality).
    
    2. REFERENCE RESOLUTION: Use chat history ONLY to identify who/what "they", "it", or "that" refers to. Once identified, create a NEW search query based on the NEW user intent.
       - User: "Were they the same nationality?" -> Search: "Nationality of A and B"
       - User: "Who they are?" -> Search: "Biography and career of A and B"

    3. HYBRID SEARCH STRATEGY:
       - Use 'hybrid_search_tool' with limit=15 to get a broad view of the entities.
       - Focus your answer on the CURRENT question's intent (e.g., identity, career, facts) while ignoring previous topics unless asked.

    RULES:
    - Cite sources as [title].
    - Answer in English. Concise and informative.
    - NEVER answer the previous question twice."""

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[trim_messages],
        checkpointer=memory,
    )
    return agent