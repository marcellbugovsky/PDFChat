# app.py

import streamlit as st
import logging
import sys
import os

# Import configuration constants
# Ensure config values are appropriate for potentially limited web server resources if deployed later
from config import (
    LOG_LEVEL, LOG_FORMAT, N_RETRIEVAL_RESULTS, DISTANCE_THRESHOLD,
    PDF_FILE_PATH # Assuming headline chunking uses these? Check document_processor
)

# Import core components
# Make sure these modules handle potential errors gracefully during initialization
try:
    from document_processor import load_and_clean_pdf_text, chunk_by_headings
    from vector_store_manager import VectorStoreManager
    from llm_handler import LLMHandler
except ImportError as e:
    st.error(f"Failed to import necessary modules: {e}. Please ensure all backend files exist and dependencies are installed.")
    st.stop() # Stop the app if core components can't be imported

# --- Logging Configuration ---
# Configure logging for the Streamlit app
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Caching Backend Components ---
# Use Streamlit's caching to load models and managers only once
@st.cache_resource
def load_vector_manager():
    """Loads the VectorStoreManager instance."""
    logger.info("Attempting to load VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        logger.info("VectorStoreManager loaded successfully.")
        return manager
    except Exception as e:
        logger.error(f"Failed to load VectorStoreManager: {e}", exc_info=True)
        st.error(f"Error initializing Vector Store: {e}. Please check model paths and ChromaDB setup.")
        return None

@st.cache_resource
def load_llm_handler():
    """Loads the LLMHandler instance."""
    logger.info("Attempting to load LLMHandler...")
    try:
        handler = LLMHandler()
        logger.info("LLMHandler loaded successfully.")
        return handler
    except Exception as e:
        logger.error(f"Failed to load LLMHandler: {e}", exc_info=True)
        st.error(f"Error initializing LLM: {e}. Check model path ({LLM_MODEL_PATH}) and llama-cpp-python setup.")
        return None

# --- Initialization and Indexing ---
# This part runs once when the app starts or when cache is cleared
# Initialize managers using cached functions
vector_manager = load_vector_manager()
llm_handler = load_llm_handler()

# Flag to indicate if initialization failed
initialization_failed = not vector_manager or not llm_handler

# Perform indexing only if needed and components are loaded
if not initialization_failed and vector_manager.needs_indexing():
    st.warning("Vector store is empty. Indexing PDF document... This might take a moment.")
    logger.info("Vector store needs indexing. Processing PDF document.")
    pdf_text = load_and_clean_pdf_text(PDF_FILE_PATH)
    if pdf_text:
        text_chunks = chunk_by_headings(pdf_text) # Using headline chunking
        if text_chunks:
            with st.spinner("Indexing document..."):
                success = vector_manager.index_documents(text_chunks)
            if success:
                st.success("Document indexed successfully!")
                logger.info("Document processed and indexed successfully.")
            else:
                st.error("Failed to index document. Check logs.")
                initialization_failed = True # Mark as failed if indexing fails
        else:
            st.error("Failed to create text chunks from PDF.")
            initialization_failed = True
    else:
        st.error(f"Failed to load text from PDF: {PDF_FILE_PATH}. Cannot index.")
        initialization_failed = True
elif not initialization_failed:
    logger.info("Using existing vector store.")


# --- Streamlit UI ---

st.title("📄 PDF Chatbot")
st.markdown("Ask questions about the content of the loaded PDF document.")

# Display error and stop if initialization failed
if initialization_failed:
    st.error("Chatbot initialization failed. Please check the logs and configuration, then restart the app.")
    st.stop()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input using chat_input
if user_query := st.chat_input("Ask your question here:"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process query and generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for the thinking indicator and final response
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve relevant chunks
                logger.info(f"Retrieving chunks for query: {user_query}")
                relevant_docs, distances = vector_manager.retrieve_relevant_chunks(
                    user_query, N_RETRIEVAL_RESULTS
                )

                # 2. Check relevance and generate answer
                answer = ""
                if relevant_docs and distances[0] <= DISTANCE_THRESHOLD:
                    logger.info(f"Relevant chunks found (best distance: {distances[0]:.4f}). Generating answer...")
                    answer = llm_handler.generate_answer(user_query, relevant_docs)
                    logger.info("Answer generated successfully.")
                elif relevant_docs:
                    logger.warning(f"Chunks found, but best distance ({distances[0]:.4f}) > threshold ({DISTANCE_THRESHOLD}).")
                    answer = "I found some related information, but it might not be relevant enough to answer your question precisely."
                else:
                    logger.info("No relevant chunks found for the query.")
                    answer = "I could not find relevant information in the document to answer that question."

                # 3. Display the final answer
                message_placeholder.markdown(answer)

            except Exception as e:
                 logger.error(f"Error processing query '{user_query}': {e}", exc_info=True)
                 message_placeholder.error("An error occurred while processing your question.")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})