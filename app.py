import streamlit as st

st.set_page_config(page_title="Multi-hop Agent", layout="wide")

from agent_setup import get_agent_executor

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Multi-hop Agent")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

agent = get_agent_executor()

if query := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        response = agent.invoke({
            "messages": st.session_state.messages
        })

        answer = response["messages"][-1].content
        st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })