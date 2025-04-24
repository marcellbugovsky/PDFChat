# config.py

# --- Pfade und Namen ---
PDF_FILE_PATH = "document.pdf" # Pfad zur Eingabe-PDF
LLM_MODEL_PATH = "./phi-3-mini-4k-instruct-q4_k_m.gguf" # Pfad zum GGUF LLM Modell
CHROMA_PERSIST_DIR = "./chroma_db" # Verzeichnis für die Vektordatenbank
CHROMA_COLLECTION_NAME = "pdf_chunks" # Name der Sammlung in ChromaDB

# --- Embedding Einstellungen ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Chunking Einstellungen ---
CHUNK_SIZE = 400 # Zielgröße für Text-Chunks (Zeichen)
CHUNK_OVERLAP = 50 # Überlappung zwischen Chunks (Zeichen)

# --- Retrieval Einstellungen ---
N_RETRIEVAL_RESULTS = 3 # Anzahl der abzurufenden Chunks
DISTANCE_THRESHOLD = 1.5 # Maximal akzeptierte Distanz für Relevanz (Anpassen!)

# --- LLM Lade-Einstellungen ---
LLM_N_CTX = 4096 # Kontextlänge für das LLM
LLM_N_THREADS = 8  # Anzahl CPU-Threads für LLM (Anpassen an Ihr System)
LLM_N_GPU_LAYERS = 0 # Anzahl der auf die GPU auszulagernden Schichten (0 für CPU)

# --- LLM Generierungs-Einstellungen ---
LLM_MAX_NEW_TOKENS = 300 # Maximale Token für die Antwort
LLM_TEMPERATURE = 0.5 # Kreativität der Antwort (niedriger für Fakten)
LLM_STOP_TOKENS = ['<|end|>', 'Antwort:', '\nFrage:'] # Wann soll das LLM aufhören zu generieren

# --- Logging ---
LOG_LEVEL = "INFO" # Loglevel (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'