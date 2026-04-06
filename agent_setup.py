from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools
from core_utils import get_resources

# Unpack 4 biến từ core_utils
_, _, _, langchain_llm = get_resources()

# Khởi tạo bộ nhớ Short-term cho Agent
# InMemorySaver giúp agent "nhớ" lịch sử trong cùng một phiên làm việc (thread_id)
memory = InMemorySaver()

def get_agent_app():
    """Tạo Agent App dưới dạng Graph."""
    
    system_message = """You are a retrieval-only QA assistant using DeepSeek Reasoner.

    CRITICAL RULES:
    1. TOOLS: You MUST call 'hybrid_search_tool' to get information before answering.
    2. MEMORY: You can see previous chat history. If the user refers to past context, use 'rewrite_query_tool'.
    3. SEARCH STRATEGY:
       - For general questions (summary, list, overview): set top_k=15.
       - For specific questions: set top_k=5.
    4. NO HALLUCINATION: Only answer using information from the tools. If not found, say "I don't know based on the database."
    5. CITATION: Use [title] for every claim.

    Answer in English. Be precise and clear.
    """

    # create_react_agent tự động xử lý vòng lặp Tool-Calling
    app = create_react_agent(
        model=langchain_llm,
        tools=tools,
        state_modifier=system_message,
        checkpointer=memory
    )
    
    return app