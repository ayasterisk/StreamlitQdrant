from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools
from core_utils import get_resources

# SỬA LỖI TẠI ĐÂY: Unpack 4 biến
_, _, _, langchain_llm = get_resources()

# Khởi tạo Short-term Memory
memory = InMemorySaver()

def get_agent_app():
    # Prompt tối ưu cho Reasoning Model (DeepSeek)
    system_message = """You are a strict retrieval-only QA assistant.

    CORE RULES:
    1. MEMORY: You can see previous messages. Use them with `rewrite_query_tool` if the current question refers to them (e.g. "tell me more about that").
    2. SEARCH: Always call `hybrid_search_tool`. 
       - If the user asks a broad/summary question, set `top_k=15`.
       - For specific questions, set `top_k=5`.
    3. FALLBACK: Use `hop2_expansion_tool` if you find titles but the text is missing details.
    4. KNOWLEDGE: NEVER use your own knowledge. If it's not in the database, say you don't know.
    5. CITATION: Cite every claim using [title].

    Answer in English. Be precise.
    """

    # Tạo Agent App với LangGraph (Thay thế AgentExecutor)
    app = create_react_agent(
        model=langchain_llm,
        tools=tools,
        state_modifier=system_message,
        checkpointer=memory
    )
    return app