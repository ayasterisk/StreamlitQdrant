from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from core_utils import get_resources
from tools_library import tools
from trim import trim_messages

_, _, _, _, langchain_llm = get_resources()


def get_agent_executor():

    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt = """You are a retrieval-only QA assistant.

        CRITICAL RULE:
        - You are ONLY allowed to answer using information retrieved from Qdrant.
        - You MUST NOT use your own knowledge.

        CONTEXT UNDERSTANDING:
        - Chat history is provided.
        - You MUST interpret follow-up questions using context.
        - Resolve references like "they", "he", "she", "it" when needed.

        QUERY STRATEGY:
        - If the question is already clear → use it directly
        - If the question is ambiguous (e.g., "they", "it", "that") 
            → infer the full meaning using chat history
        - Only rewrite the query when necessary

        WORKFLOW:
        1. Understand the question using chat history
        2. If needed, reformulate it into a clearer query
        3. Call hybrid_search_tool
        4. If enough info → answer
        5. If not → call hop2_expansion_tool

        STRICT RULES:
        - NEVER answer without calling hybrid_search_tool
        - NEVER hallucinate
        - If no data → say:
        "I don't know based on the database."

        EVIDENCE RULE:
        - MUST cite sources using [title]

        Answer in English. Be concise.
        """,
        middleware=[trim_messages],
        checkpointer=InMemorySaver(),
    )

    return agent