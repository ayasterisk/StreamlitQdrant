from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from tools_library import tools
from core_utils import get_resources

# Lấy LLM từ core_utils (giả sử langchain_llm là ChatOpenAI instance)
_, _, _, _, langchain_llm = get_resources()

# 1. Khởi tạo bộ nhớ Short-term (Checkpointer)
memory = InMemorySaver()

def get_agent_app():
    """
    Tạo Agent dưới dạng một Graph có hỗ trợ Memory.
    """
    
    system_message = """You are a retrieval-only QA assistant using DeepSeek's reasoning power.

    CRITICAL RULES:
    1. ONLY use information from tools. NEVER use internal knowledge.
    2. Short-term Memory: You can see previous conversation turns. Use them to understand context.
    3. Workflow:
       - If query is vague: call `rewrite_query_tool`.
       - ALWAYS call `hybrid_search_tool` first.
       - DYNAMIC LIMIT: If the user asks for a summary or a broad topic, set `top_k` to 10 or 15. If specific, use 5.
       - If the first results are insufficient: use `hop2_expansion_tool` with titles found.
    4. If no information is found in the database, say: "I don't know based on the database."
    5. Citing: Use [title] for every claim.
    
    Language: English. Be concise.
    """

    # 2. Khởi tạo Agent (React style)
    # create_react_agent tự động xử lý prompt, tools và agent_scratchpad
    agent = create_react_agent(
        langchain_llm, 
        tools, 
        state_modifier=system_message,
        checkpointer=memory
    )
    
    return agent

# Cách sử dụng trong main/streamlit:
# app = get_agent_app()
# config = {"configurable": {"thread_id": "user_1"}}
# for chunk in app.stream({"messages": [("user", "What is X?")]}, config):
#     print(chunk)