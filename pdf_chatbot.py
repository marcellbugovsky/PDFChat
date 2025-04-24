# pdf_chatbot.py

import pypdf
import logging
import re
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
PDF_FILE_PATH = "document.pdf"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "pdf_chunks"

def load_and_clean_pdf_text(pdf_path: str) -> str:
    """Loads text from a PDF file and performs basic cleaning."""
    logging.info(f"Loading PDF from: {pdf_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                else:
                    logging.warning(f"Could not extract text from page {page_num + 1}.")
            except Exception as e:
                logging.error(f"Error extracting text from page {page_num + 1}: {e}")
        logging.info(f"Successfully loaded {len(reader.pages)} pages.")

        # Basic Cleaning
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text
    except FileNotFoundError:
        logging.error(f"Error: PDF file not found at {pdf_path}")
        return ""
    except Exception as e:
        logging.error(f"Error loading PDF file: {e}", exc_info=True)
        return ""

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Splits text into chunks of specified size with overlap."""
    if not text: return []
    if chunk_overlap >= chunk_size: raise ValueError("chunk_overlap must be smaller than chunk_size")
    logging.info(f"Chunking text...")
    chunks = []
    start_index = 0
    text_len = len(text)
    while start_index < text_len:
        end_index = min(start_index + chunk_size, text_len) # Ensure end_index doesn't exceed text length
        chunks.append(text[start_index:end_index])
        next_start_index = start_index + chunk_size - chunk_overlap
        if next_start_index <= start_index: # Prevent infinite loops if overlap is too large or chunk_size too small
             break
        start_index = next_start_index

    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

def embed_and_store_chunks(chunks: list[str], collection_name: str, embedding_model_name: str, persist_directory: str):
    """Embeds text chunks and stores them in ChromaDB."""
    if not chunks:
        logging.warning("No chunks provided to embed and store.")
        return None

    try:
        logging.info(f"Loading embedding model: '{embedding_model_name}' (this may download the model)...")
        embedding_model = SentenceTransformer(embedding_model_name)
        logging.info("Embedding model loaded.")

        logging.info(f"Creating/loading ChromaDB collection '{collection_name}' at '{persist_directory}'")
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)

        # Get or create the collection. Clear it first for simple re-runs during development.
        try:
             logging.warning(f"Attempting to delete existing collection '{collection_name}' for re-indexing.")
             client.delete_collection(name=collection_name)
        except Exception as e:
             logging.info(f"Collection '{collection_name}' did not exist or couldn't be deleted (safe to ignore): {e}")

        collection = client.create_collection(name=collection_name)

        logging.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=True) # Show progress bar for embedding
        logging.info("Embeddings generated.")

        # Prepare data for ChromaDB
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

        logging.info(f"Adding {len(chunks)} chunks to ChromaDB collection...")
        collection.add(
            embeddings=embeddings.tolist(), # ChromaDB expects lists
            documents=chunks,
            ids=chunk_ids
        )
        logging.info("Chunks added to ChromaDB successfully.")
        return collection
    except Exception as e:
        logging.error(f"Failed to embed and store chunks: {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Clean Text
    document_text = load_and_clean_pdf_text(PDF_FILE_PATH)

    if document_text:
        # 2. Chunk Text
        text_chunks = chunk_text(document_text)

        # 3. Embed and Store Chunks in ChromaDB
        vector_store = embed_and_store_chunks(
            chunks=text_chunks,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )

        if vector_store:
             print(f"\nSuccessfully embedded and stored {vector_store.count()} chunks in ChromaDB.")
             print(f"Vector store data persisted in: {CHROMA_PERSIST_DIR}")
        else:
             print("\nFailed to create or populate the vector store.")

        print("\n--- Processing Complete ---")
    else:
        print(f"Could not load text from {PDF_FILE_PATH}. Please check the file path and format.")