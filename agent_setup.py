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

    system_prompt = """You are a highly efficient RAG assistant.
    DeepSeek Reasoner is powerful but slow, so you must minimize tool calls.

    STRATEGY:
    1. BROAD SEARCH: For questions about a topic, series, or general summary, call 'hybrid_search_tool' with a high 'limit' (15-20). 
       This allows you to get ALL necessary data in ONE single step.
    
    2. DATA ANALYSIS: Once you receive the search results (up to 20 snippets), analyze them thoroughly. 
       - If you have enough info to answer the question, STOP and answer immediately.
       - Do NOT call tools again for minor details unless essential.

    3. STANDALONE QUERY: Always rewrite the search query to be descriptive, resolving any references from chat history.

    RULES:
    - Never call tools more than twice with the same arguments.
    - Cite sources as [title].
    - Answer in English. Concise and professional."""

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[trim_messages],
        checkpointer=memory,
    )
    return agent