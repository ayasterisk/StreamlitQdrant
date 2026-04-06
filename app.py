import streamlit as st
import uuid
from agent_setup import get_agent_executor
from langchain_core.messages import AIMessage

# =========================
# 🔹 PAGE CONFIG (PHẢI ĐẦU TIÊN)
# =========================
st.set_page_config(page_title="Multi-hop Agent", layout="wide")

# =========================
# 🔹 INIT SESSION
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# =========================
# 🔹 CLEAR CHAT BUTTON
# =========================
col1, col2 = st.columns([8, 2])

with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())  # reset memory
        st.rerun()

# =========================
# 🔹 LOAD AGENT
# =========================
agent = get_agent_executor()

# =========================
# 🔹 DISPLAY CHAT
# =========================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# =========================
# 🔹 USER INPUT
# =========================
if query := st.chat_input("Ask something..."):
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").write(query)

    # =========================
    # 🔹 AGENT RESPONSE
    # =========================
    with st.chat_message("assistant"):
        response = agent.invoke(
            {
                "messages": st.session_state.messages
            },
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
        )

        # 🔥 Extract final answer từ LangGraph
        messages = response.get("messages", [])
        final_answer = ""

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        st.write(final_answer)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })