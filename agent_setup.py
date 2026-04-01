from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core_utils import get_resources
from tools_library import tools

# Get LLM
_, _, _, _, langchain_llm = get_resources()


def get_agent_executor(memory):

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a retrieval-only QA assistant.

CRITICAL RULE:
- You are ONLY allowed to answer using information retrieved from Qdrant.
- You MUST NOT use your own knowledge.

WORKFLOW:

Step 0 (Query Rewrite - OPTIONAL):
- If the question is ambiguous or depends on chat history:
  → Call rewrite_query_tool
  → If result != "ORIGINAL", use rewritten query
- Otherwise, use original query

Step 1 (MANDATORY):
- You MUST call hybrid_search_tool using the final query

Step 2 (Decision):
- If retrieved documents contain enough information → answer
- If NOT → go to Step 3

Step 3 (Fallback - Multi-hop):
- Extract titles from retrieved documents
- Call hop2_expansion_tool with those titles

STRICT RULES:
- NEVER answer without calling hybrid_search_tool
- NEVER answer if documents list is empty
- NEVER use external knowledge
- If insufficient data → say:
  "I don't know based on the database."

EVIDENCE RULE:
- Answer MUST be based ONLY on retrieved documents
- MUST cite sources using [title]

EFFICIENCY:
- Do NOT call hybrid_search_tool more than once
- Use hop2 only if necessary

Answer in English. Be concise and precise.
"""),

        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(
        langchain_llm,
        tools,
        prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )