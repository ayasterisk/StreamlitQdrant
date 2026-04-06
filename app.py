import streamlit as st
from agent_setup import get_agent_app
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Functional RAG Agent", layout="wide")

if "agent" not in st.session_state:
    st.session_state.agent = get_agent_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🚀 DeepSeek Functional Agent")

# Hiển thị hội thoại
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        # thread_id kích hoạt InMemorySaver (Short-term Memory)
        config = {"configurable": {"thread_id": "user_unique_123"}}
        
        # Invoke cực kỳ đơn giản
        result = st.session_state.agent.invoke(
            {"messages": [HumanMessage(content=prompt)]}, 
            config
        )
        
        answer = result["messages"][-1].content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})