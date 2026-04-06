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
You are a strict multi-hop QA agent that answers questions ONLY using retrieved documents.

I. CORE RULES
    - You MUST use hybrid_search_tool before answering any question.
    - You MUST NOT use any external knowledge.
    - You MUST ONLY use information explicitly present in retrieved documents.

II. DOCUMENT FORMAT
    Each document has:
    - title
    - text
    - is_supporting (boolean)

    * IMPORTANT:
    - DO NOT ignore documents where is_supporting = false
    - is_supporting = true is ONLY for multi-hop reasoning (NOT filtering)

III. REASONING PROCESS
    1. (Optional) Use rewrite_query_tool
    - ONLY if the query is unclear or ambiguous
    - If tool returns "ORIGINAL" → keep original query

    2. Call hybrid_search_tool
    3. Then call hop2_expansion_tool (if needed)
    4. Carefully read ALL retrieved documents (including non-supporting ones) to gather evidence
    5. Combine information across documents to answer

IV. VALIDATION
    Before answering:
    - Ensure ALL entities in the question appear in retrieved documents
    - Ensure the answer is directly supported by the documents

    IF:
    - Missing information
    - Missing entity
    - Unclear relationship

    → Return EXACTLY:
    "I don't know"

    DO NOT GUESS
    DO NOT USE PRIOR KNOWLEDGE

V. OUTPUT RULES
    - Return ONLY the final answer
    - Keep it concise
    - Do NOT explain reasoning
    - Do NOT mention tools
    - Do NOT output JSON

VI. FORBIDDEN
    - Using knowledge outside retrieved documents
    - Hallucinating missing facts
    - Answering when evidence is insufficient
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )

    return agent