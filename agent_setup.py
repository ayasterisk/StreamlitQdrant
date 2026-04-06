from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools
from core_utils import get_resources

# Lấy tài nguyên (4 biến từ core_utils của bạn)
_, _, _, langchain_llm = get_resources()

# Khởi tạo bộ nhớ Short-term
memory = InMemorySaver()

def get_agent_app():
    # Prompt hệ thống cho Agent
    system_instruction = """You are a retrieval-only QA assistant using DeepSeek Reasoner.

    STRICT RULES:
    1. SEARCH: You MUST call 'hybrid_search_tool' for every question.
    2. DYNAMIC LIMIT: 
       - If the question is broad (summary, list all), set top_k=15.
       - If specific facts, set top_k=5.
    3. MEMORY: Use 'rewrite_query_tool' if the user refers to past context.
    4. KNOWLEDGE: Answer ONLY using retrieved data. If not found, say you don't know.
    5. CITATION: Cite using [title].

    Answer in English. Be concise.
    """

    # state_modifier chỉ hoạt động trên langgraph bản mới
    app = create_react_agent(
        model=langchain_llm,
        tools=tools,
        state_modifier=system_instruction,
        checkpointer=memory
    )
    return app