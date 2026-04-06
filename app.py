import streamlit as st
st.set_page_config(page_title="Multi-hop Agent", layout="wide")

import uuid
from agent_setup import get_agent_executor
# =========================
# 🔹 INIT SESSION
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

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
    # Lưu user message
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
                    "thread_id": st.session_state.thread_id   # 🔥 FIX QUAN TRỌNG
                }
            }
        )

        # 🔥 Lấy output đúng format LangGraph
        if isinstance(response, dict):
            output = response.get("output", str(response))
        else:
            output = str(response)

        st.write(output)

    # Lưu assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": output
    })