"""LangGraph + Grok 챗봇"""

from typing import Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
import os

# Grok LLM 설정
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # ✅ 빠르고 좋음!
    api_key=os.getenv("XAI_API_KEY"),
    temperature=0.7
)

class State(MessagesState):
    messages: Annotated[list, add_messages]

async def call_grok(state: State) -> Dict[str, Any]:
    """Grok LLM 호출"""
    messages = state["messages"]
    
    # Grok에게 질문
    response = await llm.ainvoke(messages)
    
    return {"messages": [response]}

# 그래프 정의
graph = (
    StateGraph(State)
    .add_node("grok", call_grok)
    .add_edge("__start__", "grok")
    .add_edge("grok", "__end__")
    .compile(name="Grok Chatbot")
)