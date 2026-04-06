from langgraph.prebuilt import create_react_agent # Bản chất create_agent mới là wrapper cao cấp của cái này
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from tools_library import tools
from core_utils import get_resources

# 1. Lấy LLM từ tài nguyên (4 biến)
_, _, _, langchain_llm = get_resources()

# 2. Định nghĩa Middleware Trimming (Xử lý bộ nhớ ngắn hạn)
def middleware_manager(state):
    """
    Tự động cắt tỉa hội thoại: Giữ System Prompt + 4 tin nhắn gần nhất.
    Giúp DeepSeek không bị 'ngợp' bởi lịch sử quá dài.
    """
    messages = state["messages"]
    if len(messages) <= 6:
        return None # Không cần cắt
    
    # Giữ instruction đầu tiên và 4 context gần nhất
    trimmed = [messages[0]] + messages[-4:]
    
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + trimmed
    }

# 3. Tạo Agent theo phong cách Khai báo (Declarative)
# Gọn, nhẹ, đúng ý bạn: Model + Tools
def get_agent_app():
    system_instruction = """You are a retrieval-only assistant.
    - Always call hybrid_search_tool.
    - Use top_k=15 for broad topics, top_k=5 for specific facts.
    - Answer ONLY from tools. Cite [title]."""

    # Đây là 'create_agent' phiên bản ổn định nhất hiện nay
    agent = create_react_agent(
        model=langchain_llm,
        tools=tools,
        state_modifier=system_instruction, # Chèn System Prompt
        checkpointer=InMemorySaver(),      # Tự động hóa Short-term Memory
        # debug=True # Bật cái này nếu muốn xem nó gọi tool thế nào
    )
    return agent