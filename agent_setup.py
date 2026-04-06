import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools


def get_agent_executor():
    llm = ChatOpenAI(
        model="deepseek-reasoner",
        temperature=0,
        openai_api_key=st.secrets["DEEPSEEK_API_KEY"], 
        openai_api_base="https://api.deepseek.com/v1"
    )

    memory = InMemorySaver()

    system_prompt = """
    You are a multi-hop QA agent.

    Behavior:
    - Use rewrite_query_tool if query unclear
    - ALWAYS use hybrid_search_tool before answering
    - Use hop2_expansion_tool if needed
    - Combine results from multiple steps

    Output:
    - Final answer must be concise
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )

    return agent