from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools
from core_utils import get_resources

# Lấy resources (Unpack 4 biến theo core_utils của bạn)
_, _, _, langchain_llm = get_resources()

# Bộ nhớ Short-term
memory = InMemorySaver()

def get_agent_app():
    # Prompt cho Agent
    system_prompt = """You are a retrieval-only QA assistant using DeepSeek-Reasoner.

    RULES:
    1. SEARCH: You MUST call 'hybrid_search_tool' for every question.
    2. DYNAMIC LIMIT: 
       - If the question is broad (summary, list all, overview), set top_k=15.
       - If specific, set top_k=5.
    3. REWRITE: Use 'rewrite_query_tool' if the user's question depends on chat history.
    4. NO KNOWLEDGE: Answer ONLY based on retrieved docs. If not found, say you don't know.
    5. CITATION: Use [title] for every fact.

    Language: English. Be concise.
    """

    # Sửa từ state_modifier -> messages_modifier
    app = create_react_agent(
        model=langchain_llm,
        tools=tools,
        messages_modifier=system_prompt,
        checkpointer=memory
    )
    return app