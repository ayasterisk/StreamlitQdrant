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

### OPERATIONAL STRATEGY:
1. **Tool Selection**: You choose tools based on their "Description" and "When to use" logic. You are NOT restricted to a fixed order of steps.
2. **Contextual Thinking**: Before calling any tool, you MUST resolve pronouns (he, she, it, that company) by looking at the conversation history and replacing them with full entity names.
3. **Adaptive Search**:
    - If a search returns `STATUS: NOT_FOUND`, refine your query or try a broader term.
    - If the query is complex or involves multiple people/entities, increase `prefetch_limit` (e.g., 40-80) for better depth.
    - Use `hop2_expansion_tool` to bridge gaps between known entities and unknown facts.
4. **Error Recovery**: If you receive `STATUS: ERROR`, read the `RAISES` field to understand your mistake (e.g., invalid arguments) and fix it in your next attempt.

### CONSTRAINTS:
- Use ONLY facts from tool outputs. If the answer is not there, say "I don't know".
- No internal reasoning, JSON, or meta-commentary in the final response.
- Be direct and concise.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )

    return agent