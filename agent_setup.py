from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools
from core_utils import get_resources

# Unpack 4 biến
_, _, _, langchain_llm = get_resources()

# Khởi tạo bộ nhớ
memory = InMemorySaver()

def get_agent_app():
    # Prompt hệ thống
    system_instruction = """You are a retrieval-only QA assistant using DeepSeek Reasoner.

    STRICT RULES:
    1. SEARCH: You MUST call 'hybrid_search_tool' for every question.
    2. DYNAMIC LIMIT: 
       - Summary/General questions: top_k=15.
       - Specific facts: top_k=5.
    3. MEMORY: Use 'rewrite_query_tool' if history is relevant.
    4. NO KNOWLEDGE: Answer ONLY using tools. Cite as [title].

    Answer in English.
    """

    # state_modifier CHỈ HOẠT ĐỘNG trên langgraph >= 0.2.39
    app = create_react_agent(
        model=langchain_llm,
        tools=tools,
        state_modifier=system_instruction,
        checkpointer=memory
    )
    return app