# pdf_chatbot.py

import pypdf
import logging
import re
from sentence_transformers import SentenceTransformer
import chromadb
import os
from llama_cpp import Llama
from ctransformers import AutoModelForCausalLM # << NEU: Import für LLM
import sys

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
PDF_FILE_PATH = "document.pdf"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "pdf_chunks"
N_RETRIEVAL_RESULTS = 3 # Number of relevant chunks to retrieve
LLM_MODEL_PATH = "./phi-3-mini-4k-instruct-q4_k_m.gguf" # << NEU: Pfad zur heruntergeladenen GGUF-Datei (anpassen!)
DISTANCE_THRESHOLD = 2.0 # Schwellenwert für Relevanz (evtl. anpassen)

# --- Functions ---

def generate_answer(query: str, context_chunks: list[str], llm: AutoModelForCausalLM) -> str:
    """Generiert eine Antwort mittels LLM basierend auf dem Kontext."""

    if not context_chunks:
        return "Ich konnte keine relevanten Informationen im Dokument finden, um diese Frage zu beantworten."

    # Kontext für den Prompt vorbereiten
    context_str = "\n\n---\n\n".join(context_chunks)
    logging.info(f"Context for LLM: {context_str[:300]}...") # Log first part of context

    # Prompt-Vorlage definieren (Deutsch)
    prompt_template = f"""Anweisung: Beantworte die folgende Frage ausschließlich basierend auf dem bereitgestellten Kontext. Wenn die Antwort nicht im Kontext enthalten ist, schreibe "Die Antwort ist im vorliegenden Dokument nicht enthalten.". Antworte auf Deutsch.

Kontext:
{context_str}

Frage: {query}

Antwort: """

    # Innerhalb der generate_answer Funktion:
    logging.info("Sending prompt to LLM (llama-cpp)...")
    try:
        # LLM aufrufen mit create_completion
        response_object = llm.create_completion(
            prompt=prompt_template,
            max_tokens=300,  # Wichtig: llama-cpp nutzt max_tokens statt max_new_tokens
            temperature=0.5,
            stop=['<|end|>', 'Antwort:', '\nFrage:']  # Stop-Sequenzen, evtl. anpassen
        )
        # Antwort extrahieren
        answer = response_object['choices'][0]['text'].strip()
        logging.info("LLM response received (llama-cpp).")
        return answer
    except Exception as e:
        logging.error(f"Error during LLM generation (llama-cpp): {e}", exc_info=True)
        return "Es gab einen Fehler bei der Generierung der Antwort."
def load_and_clean_pdf_text(pdf_path: str) -> str:
    # (Function unchanged from previous step)
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
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        return full_text
    except FileNotFoundError:
        logging.error(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1) # Exit if PDF not found
    except Exception as e:
        logging.error(f"Error loading PDF file: {e}", exc_info=True)
        sys.exit(1) # Exit on other PDF loading errors

def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 25) -> list[str]:
    # (Function unchanged from previous step)
    if not text: return []
    if chunk_overlap >= chunk_size: raise ValueError("chunk_overlap must be smaller than chunk_size")
    logging.info(f"Chunking text...")
    chunks = []
    start_index = 0
    text_len = len(text)
    while start_index < text_len:
        end_index = min(start_index + chunk_size, text_len)
        chunks.append(text[start_index:end_index])
        next_start_index = start_index + chunk_size - chunk_overlap
        if next_start_index <= start_index: break
        start_index = next_start_index
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

def get_or_create_vector_store(chunks: list[str], collection_name: str, embedding_model: SentenceTransformer, persist_directory: str) -> chromadb.Collection | None:
    """Gets or creates a ChromaDB collection, indexing chunks if needed."""
    try:
        logging.info(f"Initializing ChromaDB client at '{persist_directory}'")
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)

        logging.info(f"Getting or creating collection: '{collection_name}'")
        # Check if collection exists and has items. If not, create and populate.
        try:
             collection = client.get_collection(name=collection_name)
             count = collection.count()
             logging.info(f"Found existing collection '{collection_name}' with {count} items.")
             if count == 0 and chunks:
                 logging.warning(f"Collection exists but is empty. Re-populating.")
                 client.delete_collection(name=collection_name) # Delete empty collection
                 collection = client.create_collection(name=collection_name)
                 needs_indexing = True
             elif not chunks:
                 needs_indexing = False # No chunks to index
             else:
                 # Optional: Add more sophisticated check here if PDF content changed
                 logging.info("Using existing populated collection.")
                 needs_indexing = False

        except Exception: # Collection likely doesn't exist
            logging.info(f"Collection '{collection_name}' not found. Creating and populating.")
            collection = client.create_collection(name=collection_name)
            needs_indexing = True

        if needs_indexing and chunks:
            logging.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = embedding_model.encode(chunks, show_progress_bar=True)
            logging.info("Embeddings generated.")
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            logging.info(f"Adding {len(chunks)} chunks to ChromaDB collection...")
            collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                ids=chunk_ids
            )
            logging.info("Chunks added to ChromaDB successfully.")
        elif needs_indexing and not chunks:
             logging.warning("Indexing needed but no chunks were provided.")


        return collection
    except Exception as e:
        logging.error(f"Failed to get or create vector store: {e}", exc_info=True)
        return None

def retrieve_relevant_chunks(query: str, collection: chromadb.Collection, embedding_model: SentenceTransformer, n_results: int) -> tuple[list[str], list[float]]:
    """Retrieves the most relevant text chunks from ChromaDB for a given query."""
    if not query:
        return [], []
    try:
        logging.info(f"Generating embedding for query: '{query[:50]}...'") # Log start of query
        query_embedding = embedding_model.encode([query])
        logging.info("Query embedding generated.")

        logging.info(f"Querying ChromaDB for {n_results} most relevant chunks...")
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['documents', 'distances'] # Include text and distance score
        )
        logging.info("ChromaDB query complete.")

        retrieved_docs = results.get('documents', [[]])[0] # Get the list of documents
        distances = results.get('distances', [[]])[0]      # Get the list of distances

        # Filter out potential None results if any
        valid_results = [(doc, dist) for doc, dist in zip(retrieved_docs, distances) if doc is not None and dist is not None]
        retrieved_docs = [doc for doc, dist in valid_results]
        distances = [dist for doc, dist in valid_results]


        return retrieved_docs, distances

    except Exception as e:
        logging.error(f"Error during retrieval: {e}", exc_info=True)
        return [], []

# --- Main Execution ---
if __name__ == "__main__":
    # Load embedding model once
    try:
        logging.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Fatal error loading embedding model: {e}", exc_info=True)
        sys.exit(1)

    try:
        logging.info(f"Loading LLM using llama-cpp-python from: '{LLM_MODEL_PATH}'...")
        llm = Llama(
            model_path=LLM_MODEL_PATH,  # Pfad zur GGUF-Datei
            n_ctx=4096,  # Kontextlänge (wie im Beispiel)
            n_threads=8,  # Anzahl CPU-Threads (anpassen an Ihr System)
            n_gpu_layers=0  # WICHTIG: Erstmal 0 für CPU-Test! Später erhöhen, wenn GPU vorhanden.
        )
        logging.info("LLM loaded successfully using llama-cpp-python.")
    except Exception as e:
        # Hier auch den spezifischen Fehler loggen, falls die Installation fehlte
        if "DLL load failed" in str(e) or "Can't find llama backend" in str(e):
            logging.error(
                "Failed to load llama.cpp backend. Make sure llama-cpp-python is installed correctly (requires C++ compiler during setup).")
        logging.error(f"Fatal error loading LLM with llama-cpp-python: {e}", exc_info=True)
        sys.exit(1)

    # 1. Load and Chunk PDF (only if needed for indexing)
    # We'll load/chunk lazily inside get_or_create_vector_store if indexing is required
    # This assumes the PDF content doesn't change often during interactive use.
    # For simplicity, pass empty list initially, function will load if needed.
    collection = get_or_create_vector_store(
        chunks=[], # Pass empty list, will be populated if indexing needed
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_model=embed_model,
        persist_directory=CHROMA_PERSIST_DIR
    )

    # Sicherstellen, dass der Vektor-Store bereit ist
    if collection is None or collection.count() == 0:
        logging.info("Vector store is empty or failed to initialize. Attempting to load/index PDF.")
        pdf_text = load_and_clean_pdf_text(PDF_FILE_PATH)
        if pdf_text:
            text_chunks = chunk_text(pdf_text, chunk_size=400, chunk_overlap=50)  # Verwenden Sie die angepassten Werte!
            collection = get_or_create_vector_store(  # Erneut aufrufen
                chunks=text_chunks,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_model=embed_model,
                persist_directory=CHROMA_PERSIST_DIR
            )
        else:
            logging.error("Failed to load PDF text, cannot proceed.")
            sys.exit(1)

    if collection is None or collection.count() == 0:
        logging.error("Failed to initialize or populate the vector store. Exiting.")
        sys.exit(1)

    logging.info(f"Vector store ready with {collection.count()} items.")
    print("\nPDF Chatbot Initialized. Ask questions about the document.")
    print("Type 'quit' or 'exit' to stop.")

    # 2. Start Interactive Query Loop
    while True:
        user_query = input("\nIhre Frage: ")
        if user_query.lower() in ['quit', 'exit']:
            print("Beende Chatbot.")
            break
        if not user_query.strip():
            print("Bitte geben Sie eine Frage ein.")
            continue

        # 3. Retrieve relevant chunks
        relevant_docs, distances = retrieve_relevant_chunks(
            query=user_query,
            collection=collection,
            embedding_model=embed_model,
            n_results=N_RETRIEVAL_RESULTS
        )

        if relevant_docs and distances[0] <= DISTANCE_THRESHOLD:
            # Kontext ist relevant genug -> LLM aufrufen
            answer = generate_answer(user_query, relevant_docs, llm)
            print(f"\nAntwort:\n{answer}")
        elif relevant_docs:
            # Etwas gefunden, aber nicht relevant genug
            print(
                "\nAntwort:\nIch habe verwandte Informationen gefunden, aber sie scheinen nicht relevant genug zu sein, um Ihre Frage sicher zu beantworten.")
            # Optional: Chunks trotzdem anzeigen für Debugging
            # print("\n--- Gefundene (aber evtl. nicht relevante) Chunks ---")
            # for i, (doc, dist) in enumerate(zip(relevant_docs, distances)):
            #     print(f"Chunk {i+1} (Distance: {dist:.4f}):\n'{doc}'\n")
            # print("----------------------------------------------------")
        else:
            # Nichts gefunden
            print(
                "\nAntwort:\nIch konnte keine relevanten Informationen im Dokument finden, um diese Frage zu beantworten.")
