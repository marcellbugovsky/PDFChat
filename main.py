# main.py

import logging
import sys

# Import configuration constants
from config import (
    LOG_LEVEL, LOG_FORMAT, N_RETRIEVAL_RESULTS, DISTANCE_THRESHOLD,
    PDF_FILE_PATH # CHUNK_SIZE, CHUNK_OVERLAP no longer directly used here
)

# Import core components
from document_processor import load_and_clean_pdf_text, chunk_by_headings # Use headline chunking
from vector_store_manager import VectorStoreManager
from llm_handler import LLMHandler

# Configure logging based on config
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__) # Use a logger specific to this module

def run_chatbot():
    """Initializes and runs the PDF chatbot application."""
    try:
        # --- Initialization ---
        logger.info("Initializing components...")
        vector_manager = VectorStoreManager() # Handles embeddings and ChromaDB
        llm_handler = LLMHandler() # Handles LLM loading and generation
        logger.info("Components initialized successfully.")

        # --- Ensure Vector Store is Ready ---
        # Check if indexing is needed (e.g., collection is empty)
        if vector_manager.needs_indexing():
            logger.info("Vector store needs indexing. Processing PDF document.")
            # 1. Load and clean text from PDF specified in config
            pdf_text = load_and_clean_pdf_text(PDF_FILE_PATH)
            if not pdf_text:
                 logger.error("Failed to load text from PDF. Exiting.")
                 return # Exit if PDF loading failed

            # 2. Chunk text using the headline strategy
            text_chunks = chunk_by_headings(pdf_text)
            if not text_chunks:
                 logger.error("Failed to create text chunks. Exiting.")
                 return

            # 3. Index the generated chunks
            success = vector_manager.index_documents(text_chunks)
            if not success:
                 logger.error("Failed to index documents. Exiting.")
                 return
            logger.info("Document processed and indexed successfully.")
        else:
            logger.info("Using existing vector store.")

        # --- Interactive Chat Loop ---
        print("\nPDF Chatbot Initialized. Ask questions about the document.")
        print("Type 'quit' or 'exit' to stop.")

        while True:
            try:
                user_query = input("\nYour question: ")
                if user_query.lower() in ['quit', 'exit']:
                    print("Exiting chatbot.")
                    break
                if not user_query.strip():
                    print("Please enter a question.")
                    continue

                # 1. Retrieve relevant chunks
                relevant_docs, distances = vector_manager.retrieve_relevant_chunks(
                    user_query, N_RETRIEVAL_RESULTS
                )

                # 2. Check relevance and generate answer
                answer = ""
                if relevant_docs and distances[0] <= DISTANCE_THRESHOLD:
                    # Sufficiently relevant chunks found, generate answer
                    logger.info(f"Relevant chunks found (best distance: {distances[0]:.4f}). Generating answer...")
                    answer = llm_handler.generate_answer(user_query, relevant_docs)
                elif relevant_docs:
                    # Chunks found, but distance is too high (below threshold)
                    logger.warning(f"Chunks found, but best distance ({distances[0]:.4f}) > threshold ({DISTANCE_THRESHOLD}).")
                    answer = "I found some related information, but it might not be relevant enough to answer your question precisely."
                else:
                    # No relevant chunks found
                    logger.info("No relevant chunks found for the query.")
                    answer = "I could not find relevant information in the document to answer that question."

                # 3. Print the final answer
                print(f"\nAnswer:\n{answer}")

            except KeyboardInterrupt: # Allow graceful exit with Ctrl+C
                 print("\nExiting chatbot.")
                 break
            except Exception as e: # Catch other potential errors during the loop
                 logger.error(f"An error occurred during the chat loop: {e}", exc_info=True)
                 print("An unexpected error occurred. Please try again.")


    except Exception as e:
        # Catch errors during initialization
        logger.error(f"A critical error occurred during chatbot initialization: {e}", exc_info=True)
        print("A critical error occurred during startup. Please check the logs.")

if __name__ == "__main__":
    run_chatbot()