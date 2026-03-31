from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core_utils import get_resources
from tools_library import tools

# Take resources
_, _, _, _, langchain_llm = get_resources()

def get_agent_executor(memory):
    # Define the prompt template for Agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced logic research assistant powered by the DeepSeek-R1 model. 
        Your mission is to provide precise answers to complex inquiries by strategically utilizing the provided search tools.

        Workflow:
        1. If the current question relies on previous conversation context or is ambiguous, use the 'rewrite_query_tool' to generate a standalone search query.
        2. Always use the 'hybrid_search_tool' to retrieve primary factual evidence from the knowledge base.
        3. If the retrieved documents mention key entities or bridge concepts that require deeper investigation, use the 'hop2_expansion_tool' with the specific entity titles to perform a multi-hop search.
        4. Always cite your sources accurately by including the [Source Title] at the end of your findings.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tạo Agent
    agent = create_openai_tools_agent(langchain_llm, tools, prompt)
    
    # Agent Executor
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        handle_parsing_errors=True
    )