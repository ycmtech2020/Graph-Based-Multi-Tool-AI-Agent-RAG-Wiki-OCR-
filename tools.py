import os
import io
import pytesseract
from PIL import Image
from typing import List
from langchain_core.documents import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW IMPORTS for local RAG
# from langchain_community.vectorstores import Chroma
# tools.py
# RECOMMENDED:
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Keep your original embeddings
# Remove: from langchain_astradb import AstraDBVectorStore

# --- Tool Initialization (Run once on app start) ---
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  
# Embeddings Model (using the one you chose)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wikipedia Tool (Unchanged)
WIKI_API_WRAPPER = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
WIKI_TOOL = WikipediaQueryRun(api_wrapper=WIKI_API_WRAPPER)

# --- ChromaDB Configuration ---
# Chroma DB Vector Store (Initialized in main app)
CHROMA_DB_PATH = "./chroma_data"
CHROMA_VECTOR_STORE = None
COLLECTION_NAME = "pdf_rag_agent_collection"

# Simplified function signature - no credentials needed!
def initialize_vector_store(): 
    """Initializes the global Chroma Vector Store object."""
    global CHROMA_VECTOR_STORE
    
    CHROMA_VECTOR_STORE = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=EMBEDDINGS,
        collection_name=COLLECTION_NAME,
    )
    print(f"ChromaDB Vector Store Initialized locally at {CHROMA_DB_PATH}.")
    # Return the existing store as the initial retriever for consistency
    return CHROMA_VECTOR_STORE.as_retriever()


def pdf_rag_tool(file_bytes: bytes) -> str:
    """
    Inverts a PDF file, chunks it, and stores the chunks in ChromaDB.
    Returns a status message.
    """
    global CHROMA_VECTOR_STORE
    
    if CHROMA_VECTOR_STORE is None:
        return "Error: ChromaDB Vector Store not initialized."
    
    # 1. Save file temporarily (LangChain Loaders need a path or file-like object)
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)

    # 2. Load and Split Documents (Unchanged)
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs)

    # 3. Add to Vector Store (Use Chroma's add_documents)
    # NOTE: Chroma handles upserting/adding documents to the collection.
    CHROMA_VECTOR_STORE.add_documents(doc_splits)

    # 4. Clean up (Unchanged)
    os.remove(temp_pdf_path)
    
    return f"Successfully processed and indexed {len(doc_splits)} chunks from the PDF into ChromaDB."


# --- Graph Node Functions ---

def retrieve(state):
    """Node for retrieving documents from the ChromaDB Vector Store (PDF RAG)."""
    global CHROMA_VECTOR_STORE
    print("---RETRIEVE FROM PDF RAG (CHROMA)---")
    question = state["question"]
    
    if CHROMA_VECTOR_STORE is None:
        return {"documents": [Document(page_content="Error: PDF RAG (Chroma) not available.")], "question": question}
    
    try:
        # Retrieval logic that might be failing
        retriever = CHROMA_VECTOR_STORE.as_retriever()
        documents = retriever.invoke(question)
        
        # ⚠️ Check if documents came back empty (optional, but good practice)
        if not documents:
            print("Warning: Retrieval returned 0 documents.")
            
        return {"documents": documents, "question": question}
        
    except Exception as e:
        # 🟢 CRITICAL CHANGE: Return the actual error message
        error_content = f"Retrieval Failed: {type(e).__name__}: {str(e)}"
        print(error_content) # Print to terminal for debugging
        return {"documents": [Document(page_content=error_content)], "question": question}

# tools.py

# ... (Definitions for WIKI_API_WRAPPER and WIKI_TOOL) ...

def wiki_search(state):
    """Node for performing a Wikipedia search."""
    print("---WIKIPEDIA SEARCH---")
    question = state["question"]

    # Wiki search
    result = WIKI_TOOL.invoke({"query": question})
    
    # Ensure the result is formatted as a list of Documents for consistency
    wiki_result_doc = Document(page_content=result, metadata={"source": "Wikipedia Search"}) # Added metadata for clarity
    
    return {"documents": [wiki_result_doc], "question": question}

def tesseract_ocr_tool(state):
    """Node for performing OCR on an image (best tool for image-based text)."""
    print("---TESSERACT OCR TOOL---")
    
    # In a Streamlit app, the image should be passed via the state.
    # For this example, we'll assume a file path is provided in the question
    # or that the main app stores a temporary file and puts the path in state.
    
    # Simplified approach: Check for a placeholder "ocr_file_path" in the state
    ocr_file_path = state.get("ocr_file_path")
    
    if not ocr_file_path:
        return {"documents": [Document(page_content="Error: No image file path provided for OCR.")], "question": state["question"]}

    try:
        # 1. Open image
        img = Image.open(ocr_file_path)

        # 2. Binarization (setting a threshold to make it pure black and white)
        # 128 is a common threshold value
        threshold = 128 
        img = img.point(lambda x: 0 if x < threshold else 255, '1')
        
        # 3. Perform OCR
        custom_config = r'--psm 6 --oem 3'
        extracted_text = pytesseract.image_to_string(img, config=custom_config)
        
        # 4. Format result
        ocr_result_doc = Document(page_content=f"OCR Result from image at {ocr_file_path}: {extracted_text}")
        
        return {"documents": [ocr_result_doc], "question": state["question"]}
        
    except Exception as e:
        return {"documents": [Document(page_content=f"OCR Error: {e}")], "question": state["question"]}




