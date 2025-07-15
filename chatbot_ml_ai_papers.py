import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import pinecone

def get_llm():
    return ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
# Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)
# Define your index name and embedding dimension (e.g., 1536 for many OpenAI models)
index_name = "llama-text-embed-v2-index"
dimension = 1024
# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name='llama-text-embed-v2-index',
        dimension=1024,
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
# Connect to the index
index = pc.Index(index_name)
# Initialize E5 embeddings
model_name = "intfloat/e5-large-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever( search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.35} )

#===============Define Tools===============
vectorstore_tool = create_retriever_tool(
    retriever,
    name = "PineconeVectorStore",
    description = "Retrieves relevent research papers on AI/ML topics"
)
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
tavily = TavilySearchResults()
tools = [tavily,arxiv_tool,wiki_tool]

#===============define state===============
class State(TypedDict):
  messages: Annotated[list,add_messages]

#===============define nodes===============
groq_api_key = os.getenv("GROQ_API_KEY")
llm_fr_summary = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

def llm_summarizer(docs):
    if not docs:
        return "Not Found."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a research assistant helping summarize academic papers in the field of AI and Machine Learning.

Below is a collection of excerpts from the most relevant research papers based on a user's query. The most relevant and higher-ranked documents appear first.

✅ Summarize these documents in a **detailed**, **technically accurate**, and **comprehensive** manner.
✅ Focus **more on the higher-ranking documents** that appear earlier in the list.
✅ Include **important definitions, key contributions, and findings**.
✅ The summary should be useful for someone conducting technical research on this topic.

--- Start of documents ---

{context}

--- End of documents ---

Summary:"""

    summary = llm_fr_summary.invoke(prompt)
    return summary

def summarizer_node(state):
    query = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1]
    documents = retriever.get_relevant_documents(query)

    if not documents:
        print("DEBUG: No documents found → Using query directly.")
        return {"messages": state["messages"]}

    summary = llm_summarizer(documents)
    if not summary or (hasattr(summary, "content") and summary.content.lower().strip() in ["not found.", ""]):
        print("DEBUG: Summary empty or Not Found → Using query directly.")
        return {"messages": state["messages"]}

    summary_msg = SystemMessage(content=f"""
You are helping to answer a technical question. Use the following summarized context from retrieved research papers to inform your answer:

--- Context Start ---
{summary}
--- Context End ---

Now answer the user's question: {query}
""")

    state["messages"].append(summary_msg)
    state["messages"].append(HumanMessage(content=query))

    return {"messages": state["messages"]}


def chatbot(state: State):
    llm = get_llm();
    return {"messages": [llm.bind_tools(tools).invoke(state["messages"])]}

def topic_classifier_node(state: State) -> Literal["QueryFramer", "chatbot"]:
    llm = get_llm()
    query = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1]
    prompt = f"""Is the following question related to AI or Machine Learning? Answer only "yes" or "no".\n\nQuestion: {query}"""
    response = llm.invoke(prompt).content.strip().lower()
    return "QueryFramer" if response.startswith("yes") else "chatbot"

def query_framer_node(state: State):
    llm = get_llm()
    query = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1]
    prompt = f"""Rephrase the following question to make it clearer and more specific if possible, suitable for academic or technical research:\n\nQuestion: {query}\n\nRephrased::"""
    refined = llm.invoke(prompt).content.strip()
    state["messages"].append(HumanMessage(content=refined))
    return {"messages": state["messages"]}

# ======================= Graph ============================

memory = MemorySaver()
graph_builder = StateGraph(State)

graph_builder.add_node("summarizer",summarizer_node)
graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("QueryFramer",query_framer_node)

graph_builder.add_conditional_edges(START, topic_classifier_node)
graph_builder.add_edge("QueryFramer","summarizer")
graph_builder.add_edge("summarizer","chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile(checkpointer=memory)
