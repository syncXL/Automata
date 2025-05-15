from typing_extensions import Annotated,TypedDict
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
# class ChatStripper:
#     def __init__(self):
#         self.start_ind = 0
    
#     def track(self,)

class BaseState(TypedDict):
    # """State representing the state the bot conversation"""

    messages: Annotated[list, add_messages]


state = BaseState()
state = state | {"messages" :["fuck you","mr_lover","dsf"]}
state["messages"] = state["messages"][:1]
print(state)