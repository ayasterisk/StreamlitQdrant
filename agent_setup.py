from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver
from core_utils import get_resources
from tools_library import tools

_, _, _, langchain_llm = get_resources()

memory = InMemorySaver()

def get_agent_executor():

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a retrieval-only multi-hop QA assistant.

        RULES:
        - ONLY use retrieved documents
        - NEVER use external knowledge

        MEMORY:
        - You can use chat history if needed
        - Do not repeat previous answers

        WORKFLOW:

        1. If query unclear → call rewrite_query_tool
        2. ALWAYS call hybrid_search_tool
        3. If insufficient → call hop2_expansion_tool

        FAIL:
        - If no answer → say:
        "I don't know based on the database."

        FORMAT:
        - Cite sources as [title]
        - Be concise
        """),

        MessagesPlaceholder(variable_name="messages"),
    ])

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        prompt=prompt,
        checkpointer=memory
    )

    return agent