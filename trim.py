from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model


@before_model
def trim_messages(state, runtime):
    messages = state["messages"]

    if len(messages) <= 6:
        return None

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            messages[0],
            *messages[-5:]
        ]
    }