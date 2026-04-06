import streamlit as st
from agent_setup import get_agent_app

# --- UI Setup ---
st.set_page_config(page_title="Multi-hop Agent", layout="wide")
st.title("🤖 DeepSeek RAG Agent")

# Khởi tạo Agent vào session_state (Load 1 lần duy nhất)
if "agent_app" not in st.session_state:
    with st.spinner("Initializing models and database..."):
        st.session_state.agent_app = get_agent_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý chat input
if prompt := st.chat_input("Ask about HotpotQA data..."):
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant trả lời
    with st.chat_message("assistant"):
        # Định nghĩa thread_id (để duy trì Short-term Memory)
        config = {"configurable": {"thread_id": "session_user_1"}}
        
        # Chạy Agent qua LangGraph
        input_data = {"messages": [("user", prompt)]}
        
        # Chạy và lấy kết quả cuối cùng
        # Bạn có thể dùng .stream() nếu muốn hiển thị quá trình suy luận
        result = st.session_state.agent_app.invoke(input_data, config)
        
        # Phản hồi cuối cùng nằm ở tin nhắn cuối của mảng 'messages'
        full_response = result["messages"][-1].content
        
        st.markdown(full_response)
        
        # Lưu vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": full_response})