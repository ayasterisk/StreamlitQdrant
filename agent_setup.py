from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from tools_library import tools
from core_utils import get_resources

_, _, _, _, langchain_llm = get_resources()

memory = InMemorySaver()

def get_agent_app():
    system_message = """You are a retrieval-only QA assistant.
    
    RULES:
    1. Use 'rewrite_query_tool' if the user's question is unclear or refers to past context.
    2. ALWAYS call 'hybrid_search_tool' to get data. 
    3. DYNAMIC SEARCH:
       - If the question is general (e.g., "summarize", "list all"), set top_k=15.
       - If the question is specific, set top_k=5.
    4. Use 'hop2_expansion_tool' only if the first results are missing details.
    5. ONLY answer based on tool results. Cite as [title].
    6. If not found, say "I don't know based on the database."
    
    Language: English. Be precise.
    """

    agent = create_react_agent(
        model=langchain_llm,
        tools=tools,
        state_modifier=system_message,
        checkpointer=memory
    )
    return agent