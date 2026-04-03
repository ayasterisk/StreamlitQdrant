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

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt="""You are an expert RAG assistant.

        CORE RULES:
        1. CONTEXTUALIZE: Before calling hybrid_search_tool, look at the chat history. 
           If the user uses pronouns (they, he, it, that) or follow-up questions, 
           REWRITE the query to be a standalone search term.
           Example:
           - User: "Who are they?"
           - History: Discussing Ed Wood and Scott Derrickson.
           - Action: Call hybrid_search_tool(query_text="Who are Ed Wood and Scott Derrickson?")

        2. RETRIEVAL ONLY: Only answer using information from tools. 
        3. NO HALLUCINATION: If not in database, say "I don't know based on the database."
        4. CITATION: Cite as [title].

        Language: English. Concise.
        """,
        middleware=[trim_messages],
        checkpointer=memory,
    )
    return agent