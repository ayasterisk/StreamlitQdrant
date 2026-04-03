import streamlit as st

st.set_page_config(page_title="Multi-hop Agent", layout="wide")

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from agent_setup import get_agent_executor

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Controls")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

st.title("Multi-hop RAG Agent")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask about Scott Derrickson..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        agent = get_agent_executor()
        
        config = {"configurable": {"thread_id": "user_session_1"}, "callbacks": [st_callback]}

        try:
            input_data = {"messages": [{"role": "user", "content": query}]}
            
            response = agent.invoke(input_data, config)

            final_answer = response["messages"][-1].content
            st.markdown(final_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")