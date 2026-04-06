import streamlit as st

# LỆNH NÀY PHẢI ĐẶT TRƯỚC TẤT CẢ CÁC IMPORT TỪ FILE KHÁC CỦA BẠN
st.set_page_config(page_title="Multi-hop Agent", layout="wide")

# Sau đó mới import các file nội bộ
from agent_setup import get_agent_app

st.title("🤖 DeepSeek RAG Agent")

# Khởi tạo Agent vào session_state (Load 1 lần duy nhất)
if "agent_app" not in st.session_state:
    with st.spinner("Initializing AI models and database..."):
        # Lúc này get_agent_app mới bắt đầu chạy các lệnh st.cache_resource bên trong
        st.session_state.agent_app = get_agent_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý chat input
if prompt := st.chat_input("Ask about HotpotQA data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": "session_user_1"}}
        input_data = {"messages": [("user", prompt)]}
        
        # Chạy Agent và lấy kết quả
        try:
            result = st.session_state.agent_app.invoke(input_data, config)
            full_response = result["messages"][-1].content
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")