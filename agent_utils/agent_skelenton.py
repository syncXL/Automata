from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from typing_extensions import Annotated,TypedDict,Literal
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.environ["API_KEY"]

class BaseState(TypedDict):
    # """State representing the state the bot conversation"""

    messages: Annotated[list, add_messages]


class Agent:
    def __init__(self,enable_log=False):
        self.name = "Base Agent"
        self.WELCOME_MSG = ""
        self.SYSINT = ("system","")
        self.enable_log = enable_log
        self.the_config = {
            "temperature" : 0.8,
            "api_key" : API_KEY
        }
        self.state_graph_config = {
            "recursion_limit" : 1000
        }
        self.n_tool_calls = 0
        self.bot = ChatGoogleGenerativeAI(model="gemini-2.0-flash",**self.the_config)
        self.bot_network = StateGraph(BaseState)

    def build_network(self):
        self.bot_network.add_node(self.name, self.send_message)
        self.bot_network.add_node("human_node", self.prompt_human)

        self.bot_network.add_edge(START, self.name)
        self.bot_network.add_edge("tools",self.name)
        
        self.bot_network.add_conditional_edges("human_node",self.end_session)

    def send_message(self,state: BaseState) -> BaseState:
        if state["messages"]:
            message_history = [self.SYSINT] + state["messages"]
            if (isinstance(state['messages'][-1],ToolMessage)) and (self.n_tool_calls > 1):
                toolmessages = message_history[-self.n_tool_calls:]
                toolmessages = {
                    "tool_responses": [
                        {
                        "tool_call_id" : msg.tool_call_id,
                        "output" : msg.content
                        }
                    for msg in toolmessages]
                }
                message_history = message_history[:-self.n_tool_calls]
                message_history.append(toolmessages)
            output = self.bot.invoke(message_history)
        else:
            output = AIMessage(content=self.WELCOME_MSG)
        return state | {"messages" : [output]}
    
    def prompt_human(self,state: BaseState) -> BaseState:
        """Display the last model message to the user and recieve the user's input"""
        last_msg = state["messages"][-1]
        print(f"{self.name} : {last_msg.content}")
        user_input = input("User: ")
        if user_input in {"q", "quit", "exit", "goodbye"}:
            state["finished"] = True
        return state | {"messages" : [("user", user_input)]}
    
    def end_session(self,state: BaseState) -> str|Literal["__end__"]:
        """Check if the user wants to pick"""
        if state.get("finished", False):
            return END
        else:
            return self.name
        
    def start(self):
        self.compiledState = self.bot_network.compile()
        return self.compiledState.invoke(input={"messages" : []}, config=self.state_graph_config)



