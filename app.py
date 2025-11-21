# import streamlit as st
# import os
# import cassio
# from PIL import Image
# from io import BytesIO
# from tempfile import NamedTemporaryFile
# from agent_graph import build_graph, GraphState
# from tools import initialize_vector_store, pdf_rag_tool

# # --- Initialization & Setup ---

# # Load credentials from Streamlit secrets
# try:
#     GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
#     ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
#     ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
#     ASTRA_DB_ENDPOINT = st.secrets["ASTRA_DB_ENDPOINT"]
# except:
#     st.error("Missing API keys in .streamlit/secrets.toml. Please check the documentation.")
#     st.stop()

# # Set environment variables for the graph to use
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY
# os.environ["ASTRA_DB_APPLICATION_TOKEN"] = ASTRA_DB_APPLICATION_TOKEN
# os.environ["ASTRA_DB_ID"] = ASTRA_DB_ID
# os.environ["ASTRA_DB_ENDPOINT"] = ASTRA_DB_ENDPOINT


# @st.cache_resource
# def setup_agent():
#     """Initializes Cassio, Astra DB, and compiles the LangGraph."""
#     try:
#         # 1. Initialize Cassio for Astra DB connection
#         cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        
#         # 2. Initialize the PDF RAG Vector Store
#         initialize_vector_store(
#             ASTRA_DB_APPLICATION_TOKEN, 
#             ASTRA_DB_ID, 
#             ASTRA_DB_ENDPOINT
#         )
        
#         # 3. Build and Compile the LangGraph Agent
#         app = build_graph()
#         return app
#     except Exception as e:
#         st.error(f"Agent setup failed: {e}")
#         return None
# # def setup_agent():
# #     # app.py: inside setup_agent()
# #     import socket # Needed to check if connection point is reachable
# #     from cassandra.cluster import Cluster

# #     # Use environment variables or hardcode for local Cassandra
# #     LOCAL_CASSANDRA_CONTACT_POINTS = ["127.0.0.1"] # Your local IP or 'localhost'
# #     LOCAL_CASSANDRA_PORT = 9042 # Default Cassandra port
# #     LOCAL_KEYSPACE = "my_local_keyspace" # You must create this keyspace beforehand

# #     # 1. Initialize Cassio for Local Connection
# #     try:
# #         cluster = Cluster(contact_points=LOCAL_CASSANDRA_CONTACT_POINTS, port=LOCAL_CASSANDRA_PORT)
# #         # The default keyspace must exist in your local Cassandra instance
# #         session = cluster.connect(LOCAL_KEYSPACE) 
# #         cassio.init(session=session)
# #         st.success(f"Successfully connected to local Cassandra at {LOCAL_CASSANDRA_CONTACT_POINTS[0]}")
# #         # 2. Initialize the PDF RAG Vector Store
# #         # The function no longer needs to pass credentials!
# #         initialize_vector_store()
        
# #         # 3. Build and Compile the LangGraph Agent
# #         app = build_graph()
# #         return app
# #     except Exception as e:
# #         # 🛑 Single, consolidated error handling for connection or initialization failure
# #         st.error(f"Agent setup failed. Check local Cassandra/DSE status and keyspace: {e}")
# #         return None

# # Compile the agent (cached)
# agent_app = setup_agent()
# if not agent_app:
#     st.stop()


# # --- Streamlit UI ---

# st.set_page_config(page_title="Graph-Based Multi-Tool Agent", layout="wide")
# st.title("🧠 Graph-Based Multi-Tool AI Agent (RAG, Wiki, OCR)")

# # Session State Initialization
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
# if "session_id" not in st.session_state:
#     # Use a unique ID for the user's long-term Astra DB chat history
#     st.session_state["session_id"] = "streamlit_session_" + st.session_state.get("uuid", os.urandom(8).hex())


# # Sidebar for Tools

# with st.sidebar:
#     st.header("Upload Documents")
    
#     # PDF Upload Tool
#     pdf_file = st.file_uploader("Upload PDF for RAG", type=["pdf"])
#     if pdf_file:
#         with st.spinner("Processing PDF and indexing chunks..."):
#             pdf_bytes = pdf_file.read()
#             status_message = pdf_rag_tool(pdf_bytes)
#             st.success(status_message)
#             st.session_state["pdf_rag_ready"] = True
#     else:
#         st.session_state["pdf_rag_ready"] = False
#         st.info("Upload a PDF to enable RAG.")
    
#     st.markdown("---")
    
#     # Image Upload Tool (for Tesseract OCR)
#     image_file = st.file_uploader("Upload Image for OCR", type=["jpg", "jpeg", "png"])
#     if image_file:
#         # Save image temporarily and store the path in session state
#         with NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[-1]) as tmp:
#             tmp.write(image_file.read())
#             st.session_state["ocr_file_path"] = tmp.name
#         st.success(f"Image '{image_file.name}' uploaded for OCR tool.")
#     else:
#         st.session_state["ocr_file_path"] = None
#         st.info("Upload an image to enable the Tesseract OCR tool.")

#     st.markdown("---")
#     st.caption(f"Astra DB Session ID: **{st.session_state['session_id']}**")
#     st.caption("Chat history is persisted in Astra DB.")


# # --- Main Chat Interface ---

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Main chat input
# if question := st.chat_input("Ask a question to the multi-tool agent..."):
    
#     # 1. Add user message to history
#     st.session_state.chat_history.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     # 2. Prepare initial state for LangGraph
#     initial_state = GraphState(
#         question=question,
#         generation="",
#         documents=[],
#         ocr_file_path=st.session_state.get("ocr_file_path"),
#         session_id=st.session_state["session_id"]
#     )
    
#     # 3. Execute the Agent
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking... (Routing to Tool)"):
#             try:
#                 # stream_events is used to see the LangGraph steps
#                 final_state = agent_app.invoke(initial_state)
                
#                 # Get the final generation
#                 final_answer = final_state["generation"]
                
#                 # Display the answer
#                 st.markdown(final_answer)
                
#                 # OPTIONAL: Display the context used
#                 if final_state.get("documents"):
#                     with st.expander("Show Context/Source Used"):
#                         docs = final_state["documents"]
#                         source_info = "Source documents used by the tool:\n"
#                         for i, doc in enumerate(docs):
#                             source_info += f"**{i+1}. Source:** {doc.metadata.get('source', 'N/A')}\n"
#                             source_info += f"**Content Snippet:** {doc.page_content[:500]}...\n---\n"
#                         st.markdown(source_info)
                        
#             except Exception as e:
#                 error_msg = f"An error occurred during agent execution: {e}"
#                 st.error(error_msg)
#                 final_answer = error_msg


#     # 4. Add assistant message to history
#     st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# # Clean up temp OCR file path if used
# if st.session_state.get("ocr_file_path"):
#     if os.path.exists(st.session_state["ocr_file_path"]):
#         # It's better to clean up after the user navigates away or starts a new session 
#         # to ensure the OCR tool can be re-used, but for simplicity, we'll keep it simple here.
#         # In a real app, you'd want a more robust cleanup mechanism.
#         pass

import streamlit as st
import os
from tempfile import NamedTemporaryFile
# Remove cassio import

# New imports for local components
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langgraph.checkpoint.memory import InMemorySaver
from agent_graph import build_graph, GraphState
from tools import initialize_vector_store, pdf_rag_tool

# --- Initialization & Setup ---

# Load credentials from Streamlit secrets (Only need GROQ)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Missing GROQ_API_KEY in .streamlit/secrets.toml. Please check the documentation.")
    st.stop()

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    # Set to None if missing, the library will still try to work
    HF_TOKEN = None

# Set environment variable for tools.py to access
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Set environment variable for the LLM
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
# Removed all ASTRA_DB environment variables


@st.cache_resource
def setup_agent():
    """Initializes ChromaDB Vector Store and compiles the LangGraph."""
    
    # 1. Initialize the PDF RAG Vector Store (ChromaDB)
    # The function no longer needs to pass credentials!
    try:
        # ChromaDB setup (creates ./chroma_data folder)
        initialize_vector_store()
        st.success("ChromaDB Vector Store initialized locally.")
    except Exception as e:
        st.error(f"Vector Store setup failed: {e}. Check dependencies.")
        return None

    # 2. Setup the LangGraph Checkpointer (In-Memory for simplicity)
    # NOTE: This memory will be reset when the Streamlit app restarts.
    checkpointer = InMemorySaver()
    
    # 3. Build and Compile the LangGraph Agent
    # We MUST pass the checkpointer to the build_graph function now.
    app = build_graph(checkpointer=checkpointer)
    return app


# Compile the agent (cached)
agent_app = setup_agent()
if not agent_app:
    st.stop()


# --- Streamlit UI ---

st.set_page_config(page_title="Graph-Based Multi-Tool Agent", layout="wide")
st.title("🧠 Graph-Based Multi-Tool AI Agent (RAG, Wiki, OCR)")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "session_id" not in st.session_state:
    # Use a unique ID for the user's conversation session
    st.session_state["session_id"] = "streamlit_session_" + os.urandom(8).hex()


# Sidebar for Tools

with st.sidebar:
    st.header("Upload Documents")
    
    # PDF Upload Tool
    pdf_file = st.file_uploader("Upload PDF for RAG", type=["pdf"])
    if pdf_file:
        with st.spinner("Processing PDF and indexing chunks..."):
            pdf_bytes = pdf_file.read()
            # The tool now uses the global ChromaDB instance
            status_message = pdf_rag_tool(pdf_bytes) 
            st.success(status_message)
            st.session_state["pdf_rag_ready"] = True
    else:
        st.session_state["pdf_rag_ready"] = False
        st.info("Upload a PDF to enable RAG.")
    
    st.markdown("---")
    
    # Image Upload Tool (for Tesseract OCR)
    image_file = st.file_uploader("Upload Image for OCR", type=["jpg", "jpeg", "png"])
    if image_file:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[-1]) as tmp:
            tmp.write(image_file.read())
            st.session_state["ocr_file_path"] = tmp.name
        st.success(f"Image '{image_file.name}' uploaded for OCR tool.")
    else:
        st.session_state["ocr_file_path"] = None
        st.info("Upload an image to enable the Tesseract OCR tool.")

    st.markdown("---")
    st.caption(f"Session ID: **{st.session_state['session_id']}**")
    st.caption("Chat history is kept in memory (reset on app restart).")


# --- Main Chat Interface ---

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
if question := st.chat_input("Ask a question to the multi-tool agent..."):
    
    # 1. Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Prepare initial state for LangGraph
    initial_state = GraphState(
        question=question,
        generation="",
        documents=[],
        ocr_file_path=st.session_state.get("ocr_file_path"),
        session_id=st.session_state["session_id"]
    )
    
    # 3. Execute the Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Routing to Tool)"):
            try:
                # 🟢 Define the configuration required by the Checkpointer
                config = {"configurable": {"thread_id": st.session_state["session_id"]}}
                # The agent_app now uses the checkpointer initialized in setup_agent()
                final_state = agent_app.invoke(initial_state, config=config) 
                
                final_answer = final_state["generation"]
                
                st.markdown(final_answer)
                
                if final_state.get("documents"):
                    with st.expander("Show Context/Source Used"):
                        docs = final_state["documents"]
                        source_info = "Source documents used by the tool:\n"
                        for i, doc in enumerate(docs):
                            source_info += f"**{i+1}. Source:** {doc.metadata.get('source', 'N/A')}\n"
                            source_info += f"**Content Snippet:** {doc.page_content[:500]}...\n---\n"
                        st.markdown(source_info)
                        
            except Exception as e:
                error_msg = f"An error occurred during agent execution: {e}"
                st.error(error_msg)
                final_answer = error_msg


    # 4. Add assistant message to history
    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

# Clean up temp OCR file path if used
if st.session_state.get("ocr_file_path") and os.path.exists(st.session_state["ocr_file_path"]):
    # Keep it simple for Streamlit dev environment
    pass