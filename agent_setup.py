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
    You are an Autonomous, Fact-Based Research Specialist. Your goal is to provide accurate answers using ONLY retrieved data.

    ### TRACE OPTIMIZATION STRATEGY:
    - To ensure high speed, search results are provided as **Snippets** (short versions).
    - If a snippet looks relevant but is cut off (...), use `hop2_expansion_tool` with the specific Title to dive deeper.

    ### OPERATIONAL STRATEGY:
    1. **Tool Selection**: You choose tools based on their "Description". Call tools as many times as needed.
    2. **Contextual Thinking**: Before calling any tool, you MUST resolve pronouns (he, she, it) by replacing them with full entity names from history.
    3. **Adaptive Search**:
        - If `STATUS: NOT_FOUND`, pivot your query terms.
        - If `STATUS: ERROR`, fix your arguments based on the `RAISES` info.

    ### CONSTRAINTS:
    - No internal reasoning or JSON in the final answer.
    - Answer in the same language as the user's query.
    """

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return agent