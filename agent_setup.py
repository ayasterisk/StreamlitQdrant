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
        Your mission is to provide precise answers to complex inquiries by strategically and efficiently utilizing the provided search tools.

        Workflow:
        1. If the current question relies on previous context, use 'rewrite_query_tool' to create a standalone query.
        2. Use 'hybrid_search_tool' to retrieve factual evidence. **To ensure efficiency, consolidate all key entities and their relationships into a single, comprehensive search query whenever possible, rather than searching for them one by one.**
        3. Only perform additional searches if the primary search results are clearly missing critical information needed to bridge the gap between entities.
        4. Use 'hop2_expansion_tool' specifically when you have identified exact Source Titles from step 2 that require deeper detail.
        5. Always cite your sources accurately by including the [Source Title] at the end of your findings.
        6. Provide the final answer in English. Be concise and avoid redundant tool calls if the information is already sufficient."""),
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
        max_iterations=3,
        handle_parsing_errors=True
    )