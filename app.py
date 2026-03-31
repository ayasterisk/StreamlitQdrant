import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from agent_setup import get_agent_executor

# UI Setup
st.set_page_config(page_title="Multi-hop Agent", layout="wide")

# Memory & Messages Initialization
if "langchain_memory" not in st.session_state:
    st.session_state.langchain_memory = ConversationBufferWindowMemory(
        k=5, 
        memory_key="chat_history", 
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar System Info & Controls
with st.sidebar:
    st.title("System Info & Controls")
    st.success(f"Tracing: ON (LangSmith)")
    st.info(f"LLM: DeepSeek-R1")
    st.markdown("[LangSmith Dashboard](https://smith.langchain.com/)")
    
    if st.button("Delete Conversation History"):
        st.session_state.messages = []
        st.session_state.langchain_memory.clear()
        st.rerun()

# Show chat history
st.title("Multi-hop Agent")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input
if query := st.chat_input("Ask a complex question..."):
    # Save user query to session state and display
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        # Callback Handler to stream responses in real-time to Streamlit UI
        st_callback = StreamlitCallbackHandler(st.container())
        
        # Agent Executor
        agent_executor = get_agent_executor(st.session_state.langchain_memory)
        
        # Run Agent and Stream Response
        try:
            response = agent_executor.invoke(
                {"input": query}, 
                {"callbacks": [st_callback]}
            )
            
            final_answer = response["output"]
            st.markdown(final_answer)
            
            # Save assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")