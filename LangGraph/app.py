import streamlit as st
import json
from typing import Annotated
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langchain_tavily import TavilySearch
import arxiv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .agent-message {
        padding: 0.75rem; margin: 0.5rem 0;
        border-radius: 0.5rem; border-left: 4px solid;
    }
    .researcher {
        border-left-color: #4CAF50; background-color: #e8f5e9;
    }
    .writer {
        border-left-color: #2196F3; background-color: #e3f2fd;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------ STATE DEFINITION ------------------
class AgentState(MessagesState):
    messages: Annotated[list[dict], add_messages]
    next_agent: str
    research_data: str | None
    tool_round: int = 0
    tools_used: list[str] = []


# ------------------ TOOLS ------------------
def init_tools():
    tavily_client = TavilySearch(max_results=3)

    @tool
    def search_web(query: str) -> str:
        """Search the web using Tavily for current information."""
        try:
            results = tavily_client.invoke(query)
            if isinstance(results, list) and len(results) > 0:
                formatted = []
                for i, r in enumerate(results[:3], 1):
                    title = r.get('title', 'No title')
                    content = r.get('content', 'No content')
                    url = r.get('url', 'No URL')
                    formatted.append(f"{i}. {title}\n   {content[:200]}...\n   Source: {url}")
                return "\n\n".join(formatted)
            return f"No results found for '{query}'"
        except Exception as e:
            return f"Search failed: {e}"

    @tool
    def arxiv_search(query: str, max_results: int = 3) -> str:
        """Search arXiv using the arxiv package and return titles + abstracts + PDF links."""
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            parsed = []
            for r in search.results():
                title = r.title or "No title"
                summary = (r.summary or "No summary").strip()
                pdf_url = r.pdf_url or r.entry_id
                parsed.append(f"- {title}\n  {summary[:500]}...\n  {pdf_url}")
            return "\n\n".join(parsed) if parsed else "No results"
        except Exception as e:
            return f"arXiv search failed: {e}"

    return [search_web, arxiv_search]


# ------------------ LLM ------------------
@st.cache_resource
def init_llm():
    return ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")


# ------------------ AGENTS ------------------
def researcher_agent(state: AgentState):
    messages = state["messages"]
    llm = init_llm()
    tools = init_tools()
    researcher = llm.bind_tools(tools)

    system_msg = SystemMessage(
        content="You are a research assistant. If you need external data, select exactly ONE tool from [search_web, arxiv_search] and call it once."
    )

    MAX_TOOL_ROUNDS = 2
    current_round = state.get("tool_round", 0)
    tools_used = state.get("tools_used", [])

    if current_round >= MAX_TOOL_ROUNDS:
        res = llm.invoke(messages + [system_msg])
        return {"messages": [res], "research_data": res.content, "tools_used": tools_used}

    res = researcher.invoke(messages + [system_msg])
    content = getattr(res, "content", str(res))

    # Detect tool usage
    used_tool_name = None
    try:
        if hasattr(res, "tool_calls") and res.tool_calls:
            used_tool_name = res.tool_calls[0]["name"]
            tools_used.append(used_tool_name)
    except Exception:
        pass

    new_round = current_round + (1 if used_tool_name else 0)

    return {
        "messages": [res],
        "research_data": content,
        "tool_round": new_round,
        "tools_used": tools_used,
    }


def writer_agent(state: AgentState):
    messages = state["messages"]
    llm = init_llm()
    system_msg = SystemMessage(content="You are a concise academic writer. Summarize the findings clearly in markdown.")
    res = llm.invoke(messages + [system_msg])
    return {"messages": [res]}


# ------------------ WORKFLOW ------------------
def build_workflow():
    tools = init_tools()
    workflow = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    workflow.set_entry_point("researcher")
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("writer", writer_agent)

    workflow.add_conditional_edges("researcher", tools_condition, {"tools": "tools", "__end__": "writer"})
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("writer", END)

    return workflow.compile()


# ------------------ STREAMLIT UI ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔍 AI Research Assistant")
st.markdown("### Multi-Agent Research System with Web & Academic Search")

query = st.text_input("Enter your research question:", placeholder="e.g., What are the latest advancements in AI?")
submit = st.button("🚀 Start Research", type="primary")

if submit and query:
    with st.spinner("🔬 Researchers are analyzing..."):
        try:
            graph = build_workflow()
            response = graph.invoke({"messages": [HumanMessage(content=query)]})

            st.session_state.messages.append({
                "query": query,
                "response": response,  # store full response dict
            })
            st.success("✅ Research complete!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ------------------ RESULTS DISPLAY ------------------
if st.session_state.messages:
    st.markdown("---")
    st.header("📊 Results")

    latest = st.session_state.messages[-1]
    response = latest["response"]

    # ---- Main Markdown Output ----
    final_messages = response.get("messages", [])
    if final_messages:
        final_msg = final_messages[-1]
        if isinstance(final_msg, AIMessage):
            st.markdown(final_msg.content, unsafe_allow_html=True)
        else:
            st.markdown(str(final_msg))

    # ---- Collapsible Trace Viewer ----
    with st.expander("🔍 View Full Response Trace (for debugging / curiosity)"):
        st.json(response, expanded=False)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Powered by LangGraph, Groq, Tavily & arXiv</div>", unsafe_allow_html=True)
