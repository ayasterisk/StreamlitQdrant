import streamlit as st
from agent_setup import get_agent_app

# UI Setup
st.set_page_config(page_title="DeepSeek RAG Agent", layout="wide")

st.title("Multi-hop RAG Agent")

if "agent_app" not in st.session_state:
    st.session_state.agent_app = get_agent_app()

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": "streamlit_session_1"}}
        
        input_data = {"messages": [("user", prompt)]}
        
        full_response = ""
        placeholder = st.empty()
        
        result = st.session_state.agent_app.invoke(input_data, config)
        
        full_response = result["messages"][-1].content
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})