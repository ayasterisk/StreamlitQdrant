import streamlit as st

# PHẢI LÀ LỆNH ĐẦU TIÊN
st.set_page_config(page_title="DeepSeek RAG Agent", layout="wide")

from agent_setup import get_agent_app

st.title("🤖 DeepSeek Multi-hop RAG")
st.markdown("---")

# Khởi tạo Agent (chỉ load 1 lần)
if "agent_app" not in st.session_state:
    with st.spinner("Đang khởi tạo hệ thống..."):
        st.session_state.agent_app = get_agent_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý Chat Input
if prompt := st.chat_input("Nhập câu hỏi tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # thread_id để Agent duy trì bộ nhớ trong phiên này
        config = {"configurable": {"thread_id": "temp_user_01"}}
        
        try:
            # Gọi Agent (invoke)
            response_data = st.session_state.agent_app.invoke(
                {"messages": [("user", prompt)]}, 
                config
            )
            
            # Lấy tin nhắn cuối cùng từ mảng trả về
            answer = response_data["messages"][-1].content
            st.markdown(answer)
            
            # Lưu lịch sử
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {str(e)}")