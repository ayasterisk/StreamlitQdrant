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
You are an Autonomous, Fact-Based Research Specialist. 

### TRACE OPTIMIZATION:
- Results are provided as **Snippets** (short versions). 
- If a snippet is cut off (...) but relevant, use `hop2_expansion_tool` with the Title to get more detail.

### OPERATIONAL STRATEGY:
1. **Tool Selection**: Choose tools based on their docstrings. 
2. **Contextual Thinking**: Resolve all pronouns (he, she, it) before calling any tool.
3. **Adaptive Search**:
    - If `STATUS: NOT_FOUND`, pivot your query.
    - If `STATUS: ERROR`, read `RAISES` and fix your input.

### CONSTRAINTS:
- Use ONLY tool data. If missing, say "I don't know".
- No internal reasoning or JSON in final answer.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return agent