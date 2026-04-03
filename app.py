import streamlit as st

# UI
st.set_page_config(page_title="Multi-hop Agent", layout="wide")

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from agent_setup import get_agent_executor

# Init UI memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("System Info & Controls")
    st.success("Tracing: ON (LangSmith)")
    st.info("LLM: DeepSeek-R1")

    if st.button("Delete Conversation History"):
        st.session_state.messages = []
        st.rerun()

# Display history
st.title("Multi-hop Agent")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
if query := st.chat_input("Ask a complex question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):

        st_callback = StreamlitCallbackHandler(st.container())

        agent = get_agent_executor()

        config = {"configurable": {"thread_id": "1"}}

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config
            )

            final_answer = response["messages"][-1].content

            st.markdown(final_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")