import streamlit as st

st.set_page_config(page_title="Multi-hop Agent", layout="wide")

import uuid
from agent_setup import get_agent_executor
from langchain_core.messages import AIMessage

st.title("Multi-hop Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()
agent = get_agent_executor()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask something..."):
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("⏳ Thinking...")
            with st.spinner("Processing..."):
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

        messages = response.get("messages", [])
        final_answer = ""

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        placeholder.empty()

        st.write(final_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })