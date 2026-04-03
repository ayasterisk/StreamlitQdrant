from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from core_utils import get_resources
from tools_library import tools
from trim import trim_messages

# Get LLM
_, _, _, _, langchain_llm = get_resources()


def get_agent_executor():

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt="""You are a retrieval-only QA assistant.

            CRITICAL RULE:
            - You are ONLY allowed to answer using information retrieved from Qdrant.
            - You MUST NOT use your own knowledge.

            WORKFLOW:
            1. You MUST call hybrid_search_tool first
            2. If enough information → answer
            3. If not → call hop2_expansion_tool

            STRICT RULES:
            - NEVER answer without calling hybrid_search_tool
            - NEVER hallucinate
            - If no data → say:
            "I don't know based on the database."

            EVIDENCE RULE:
            - MUST cite sources using [title]

            Use chat history if needed.

            Answer in English. Be concise.
            """,
        middleware=[trim_messages],
        checkpointer=InMemorySaver(),
    )

    return agent