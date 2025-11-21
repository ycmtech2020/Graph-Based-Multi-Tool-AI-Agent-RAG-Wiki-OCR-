# from typing import List, Literal, TypedDict
# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
# from langchain_groq import ChatGroq
# from langchain_astradb import AstraDBChatMessageHistory
# from langgraph.graph import END, StateGraph, START
# from tools import retrieve, wiki_search, tesseract_ocr_tool

# # --- Graph State Definition ---

# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.
#     """
#     question: str
#     generation: str
#     documents: List[Document]
#     # To handle image files for OCR
#     ocr_file_path: str 
#     # For chat history / memory persistence
#     session_id: str


# # --- Routing Logic ---

# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant tool/datasource."""
#     datasource: Literal["pdf_rag", "wiki_search", "ocr_tool", "rag_chat"] = Field(
#         ...,
#         description="Given a user question, choose to route it to: "
#                     "1. 'pdf_rag': For questions about the content of the uploaded PDF. "
#                     "2. 'wiki_search': For general knowledge or current events. "
#                     "3. 'ocr_tool': If the question clearly asks to extract text from an image (OCR). "
#                     "4. 'rag_chat': If it's a follow-up question in the existing conversation (history).",
#     )

# def create_router(llm: ChatGroq):
#     """Creates the LLM-powered routing chain."""
    
#     structured_llm_router = llm.with_structured_output(RouteQuery)

#     system = """You are an expert router. Route the user question to the most relevant tool:
#     - Use 'pdf_rag' for questions specifically about the uploaded document content.
#     - Use 'wiki_search' for general or real-time information.
#     - Use 'ocr_tool' if the user explicitly asks to 'read', 'extract text from', or 'transcribe' an image.
#     - Use 'rag_chat' if the question is a follow-up to the current conversation (e.g., 'What was my previous question?'). 
#       Only use this if the question is conversational, not if it's a new topic relevant to a tool.
#     """
#     route_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "{question}"),
#         ]
#     )
#     return route_prompt | structured_llm_router

# def route_question(state):
#     """Route question to one of the tool nodes or the final generation node."""
#     print("---ROUTE QUESTION---")
#     question = state["question"]
#     router = create_router(state["llm"]) # LLM is passed via the state in the final setup
    
#     # We must explicitly handle the LLM passing, but for a clean graph structure, 
#     # let's modify the GraphState to include the LLM or pass the router in directly.
#     # For now, we'll assume the LLM is initialized globally or in the main script.
#     # Let's create the LLM object here for simplicity in this file.
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#     llm_router = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0)
    
#     # Re-create router with the LLM
#     structured_llm_router = llm_router.with_structured_output(RouteQuery)
#     system = "..." # same system prompt as above
#     route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}"),])
#     question_router = route_prompt | structured_llm_router
    
#     source = question_router.invoke({"question": question})
    
#     if source.datasource == "wiki_search":
#         print("---ROUTE TO WIKI SEARCH---")
#         return "wiki_search"
#     elif source.datasource == "pdf_rag":
#         print("---ROUTE TO PDF RAG RETRIEVAL---")
#         return "retrieve"
#     elif source.datasource == "ocr_tool":
#         print("---ROUTE TO TESSERACT OCR TOOL---")
#         return "ocr_tool"
#     else: # Default or 'rag_chat' goes to generation
#         print("---ROUTE TO GENERATION (CHATTING)---")
#         # In a more complex agent, 'rag_chat' might be a separate path. 
#         # Here, we treat it as the final step with the *current* context.
#         return "generate" 


# # --- Generation Node ---

# def generate(state):
#     """
#     Generate an answer using the retrieved documents and chat history.
#     """
#     print("---GENERATE ANSWER---")
#     question = state["question"]
#     documents = state["documents"]
#     session_id = state["session_id"]
    
#     # Initialize LLM with a safe default model
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#     llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.2)
    
#     # # Long-term memory (Astra DB)
#     history = AstraDBChatMessageHistory(
#         session_id=session_id,
#         token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
#         api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT"),
#         database_id=os.environ.get("ASTRA_DB_ID"),
#         collection_name="chat_history_collection"
#     )

#     # agent_graph.py: inside generate(state)
#     # Long-term memory (Local Cassandra/DSE)
#     # history = AstraDBChatMessageHistory(
#     #     session_id=session_id,
#     #     # NOTE: Omit token, api_endpoint, and database_id to use the session set by cassio.init()
#     #     collection_name="chat_history_collection" 
#     # )
    
#     # Short-term memory (just the last few turns from history)
#     # Get last 5 messages for short-term context
#     short_term_history = history.messages[-5:]
    
#     # Context
#     context = "\n\n".join([doc.page_content for doc in documents])
    
#     # Prompt with History and Context
#     prompt_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", """You are a helpful, graph-based AI assistant. 
#              Answer the user's question based on the provided context (from RAG, Wikipedia, or OCR) and chat history.
#              If the context is irrelevant or not provided, answer based on your general knowledge.
             
#              ## Relevant Context
#              {context}
             
#              ## Chat History
#              {chat_history}
#              """),
#             ("human", "{question}"),
#         ]
#     )

#     # Convert chat history for prompt
#     chat_history_str = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in short_term_history])
    
#     # RAG Chain
#     rag_chain = prompt_template | llm
    
#     generation = rag_chain.invoke(
#         {"context": context, "question": question, "chat_history": chat_history_str}
#     )
    
#     # Save current turn to long-term memory
#     history.add_user_message(question)
#     history.add_ai_message(generation.content)

#     return {"generation": generation.content, "question": question, "documents": documents}


# # --- Build Graph ---

# def build_graph():
#     """Builds and compiles the LangGraph workflow."""
#     workflow = StateGraph(GraphState)
    
#     # Define the nodes
#     workflow.add_node("retrieve", retrieve) 
#     workflow.add_node("wiki_search", wiki_search)
#     workflow.add_node("ocr_tool", tesseract_ocr_tool)
#     workflow.add_node("generate", generate)
    
#     # Conditional Edges from START (Router)
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "wiki_search": "wiki_search",
#             "retrieve": "retrieve",
#             "ocr_tool": "ocr_tool",
#             "generate": "generate", # Direct generation for conversational follow-ups
#         },
#     )
    
#     # Edges from Tool Nodes to Final Generation
#     workflow.add_edge("retrieve", "generate")
#     workflow.add_edge("wiki_search", "generate")
#     workflow.add_edge("ocr_tool", "generate")
    
#     # Final Edge
#     workflow.add_edge("generate", END)
    
#     return workflow.compile()







# from typing import List, Literal, TypedDict
# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
# from langchain_groq import ChatGroq
# # New import for local memory
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# from langgraph.graph import END, StateGraph, START
# from langgraph.checkpoint.memory import InMemorySaver # New import hint
# from tools import retrieve, wiki_search, tesseract_ocr_tool

# # --- Graph State Definition ---

# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.
#     """
#     question: str
#     generation: str
#     documents: List[Document]
#     ocr_file_path: str 
#     # For chat history / memory persistence (session_id is still useful for context)
#     session_id: str


# # --- Routing Logic (Unchanged) ---
# # ... (Keep RouteQuery and create_router functions as they are) ...

# def route_question(state):
#     # ... (Keep route_question function as it is) ...
#     pass 


# # --- Generation Node ---
# # Global in-memory storage for chat history (persists while the app runs)
# # NOTE: This is a simplification; a full production system would use the checkpointer for history.
# # Since LangGraph uses the checkpointer, we can simplify this for a local example.
# history_store = {} 

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     """Retrieves the correct chat history object for the session ID."""
#     if session_id not in history_store:
#         history_store[session_id] = InMemoryChatMessageHistory()
#     return history_store[session_id]


# def generate(state):
#     """
#     Generate an answer using the retrieved documents and chat history.
#     """
#     print("---GENERATE ANSWER---")
#     question = state["question"]
#     documents = state["documents"]
#     session_id = state["session_id"]
    
#     # Initialize LLM with a safe default model
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#     llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.2)
    
#     # Local In-Memory History
#     history = get_session_history(session_id)
    
#     # Short-term memory (just the last few turns from history)
#     short_term_history = history.messages[-5:]
    
#     # ... (Context and Prompt Template logic is UNCHANGED) ...
#     # Context
#     context = "\n\n".join([doc.page_content for doc in documents])
    
#     # Prompt with History and Context
#     prompt_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", """You are a helpful, graph-based AI assistant. 
#              Answer the user's question based on the provided context (from RAG, Wikipedia, or OCR) and chat history.
#              If the context is irrelevant or not provided, answer based on your general knowledge.
             
#              ## Relevant Context
#              {context}
             
#              ## Chat History
#              {chat_history}
#              """),
#             ("human", "{question}"),
#         ]
#     )

#     # Convert chat history for prompt
#     chat_history_str = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in short_term_history])
    
#     # RAG Chain
#     rag_chain = prompt_template | llm
    
#     generation = rag_chain.invoke(
#         {"context": context, "question": question, "chat_history": chat_history_str}
#     )
    
#     # Save current turn to long-term memory
#     history.add_user_message(question)
#     history.add_ai_message(generation.content)

#     return {"generation": generation.content, "question": question, "documents": documents}


# # --- Build Graph ---

# # The function must now accept the checkpointer argument
# def build_graph(checkpointer: InMemorySaver):
#     """Builds and compiles the LangGraph workflow."""
#     workflow = StateGraph(GraphState)
    
#     # Define the nodes
#     workflow.add_node("retrieve", retrieve) 
#     workflow.add_node("wiki_search", wiki_search)
#     workflow.add_node("ocr_tool", tesseract_ocr_tool)
#     workflow.add_node("generate", generate)
    
#     # Conditional Edges from START (Router)
#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "wiki_search": "wiki_search",
#             "retrieve": "retrieve",
#             "ocr_tool": "ocr_tool",
#             "generate": "generate",
#         },
#     )
    
#     # Edges from Tool Nodes to Final Generation
#     workflow.add_edge("retrieve", "generate")
#     workflow.add_edge("wiki_search", "generate")
#     workflow.add_edge("ocr_tool", "generate")
    
#     # Final Edge
#     workflow.add_edge("generate", END)
    
#     # Pass the checkpointer during compilation!
#     return workflow.compile(checkpointer=checkpointer)



import os
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# New import for local memory (Correct for your langchain-core 1.0.7 version)
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver # New import hint
from tools import retrieve, wiki_search, tesseract_ocr_tool

# --- Graph State Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    ocr_file_path: str 
    session_id: str


# ===============================================
# --- ROUTING LOGIC (ADDED FROM COMMENTED BLOCK) ---
# ===============================================

class RouteQuery(BaseModel):
    """Route a user query to the most relevant tool/datasource."""
    datasource: Literal["pdf_rag", "wiki_search", "ocr_tool", "rag_chat"] = Field(
        ...,
        description="Given a user question, choose to route it to: "
                    "1. 'pdf_rag': For questions about the content of the uploaded PDF. "
                    "2. 'wiki_search': For general knowledge or current events. "
                    "3. 'ocr_tool': If the question clearly asks to extract text from an image (OCR). "
                    "4. 'rag_chat': If it's a follow-up question in the existing conversation (history).",
    )

def create_router(llm: ChatGroq):
    """Creates the LLM-powered routing chain."""
    
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert router. Route the user question to the most relevant tool:
    - Use 'pdf_rag' for questions specifically about the uploaded document content.
    - Use 'wiki_search' for general or real-time information.
    - Use 'ocr_tool' if the user explicitly asks to 'read', 'extract text from', or 'transcribe' an image.
    - Use 'rag_chat' if the question is a follow-up to the current conversation (e.g., 'What was my previous question?'). 
      Only use this if the question is conversational, not if it's a new topic relevant to a tool.
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    return route_prompt | structured_llm_router

def route_question(state):
    """Route question to one of the tool nodes or the final generation node."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Initialize LLM for the router here
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm_router = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0)
    
    # Re-create router with the LLM
    structured_llm_router = llm_router.with_structured_output(RouteQuery)
    
    # Re-create the prompt for the router (simplified, since it's defined above)
    system = """You are an expert router. Route the user question to the most relevant tool:
    - Use 'pdf_rag' for questions specifically about the uploaded document content.
    - Use 'wiki_search' for general or real-time information.
    - Use 'ocr_tool' if the user explicitly asks to 'read', 'extract text from', or 'transcribe' an image.
    - Use 'rag_chat' if the question is a follow-up to the current conversation (e.g., 'What was my previous question?'). 
      Only use this if the question is conversational, not if it's a new topic relevant to a tool.
    """
    route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}"),])
    question_router = route_prompt | structured_llm_router
    
    source = question_router.invoke({"question": question})
    
    if source.datasource == "wiki_search":
        print("---ROUTE TO WIKI SEARCH---")
        return "wiki_search"
    elif source.datasource == "pdf_rag":
        print("---ROUTE TO PDF RAG RETRIEVAL---")
        return "retrieve"
    elif source.datasource == "ocr_tool":
        print("---ROUTE TO TESSERACT OCR TOOL---")
        return "ocr_tool"
    else: # Default or 'rag_chat' goes to generation
        print("---ROUTE TO GENERATION (CHATTING)---")
        return "generate" 


# ===============================================
# --- GENERATION NODE (FROM CURRENT CODE) ---
# ===============================================

# Global in-memory storage for chat history 
history_store = {} 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves the correct chat history object for the session ID."""
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]


def generate(state):
    """
    Generate an answer using the retrieved documents and chat history.
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    session_id = state["session_id"]
    
    # Initialize LLM with a safe default model
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)
    
    # Local In-Memory History
    history = get_session_history(session_id)
    
    # Short-term memory (just the last few turns from history)
    short_term_history = history.messages[-5:]
    
    # Context
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Prompt with History and Context
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful, graph-based AI assistant. 
             Answer the user's question based on the provided context (from RAG, Wikipedia, or OCR) and chat history.
             If the context is irrelevant or not provided, answer based on your general knowledge.
             
             ## Relevant Context
             {context}
             
             ## Chat History
             {chat_history}
             """),
            ("human", "{question}"),
        ]
    )

    # Convert chat history for prompt
    chat_history_str = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in short_term_history])
    
    # RAG Chain
    rag_chain = prompt_template | llm
    
    generation = rag_chain.invoke(
        {"context": context, "question": question, "chat_history": chat_history_str}
    )
    
    # Save current turn to long-term memory
    history.add_user_message(question)
    history.add_ai_message(generation.content)

    return {"generation": generation.content, "question": question, "documents": documents}


# ===============================================
# --- BUILD GRAPH (FROM CURRENT CODE) ---
# ===============================================

def build_graph(checkpointer: InMemorySaver):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve) 
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("ocr_tool", tesseract_ocr_tool)
    workflow.add_node("generate", generate)
    
    # Conditional Edges from START (Router)
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "retrieve": "retrieve",
            "ocr_tool": "ocr_tool",
            "generate": "generate",
        },
    )
    
    # Edges from Tool Nodes to Final Generation
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("wiki_search", "generate")
    workflow.add_edge("ocr_tool", "generate")
    
    # Final Edge
    workflow.add_edge("generate", END)
    
    # Pass the checkpointer during compilation!
    return workflow.compile(checkpointer=checkpointer)