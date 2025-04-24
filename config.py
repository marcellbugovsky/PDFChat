# config.py

# --- Paths and Names ---
PDF_FILE_PATH: str = "document.pdf" # Path to the input PDF document
LLM_MODEL_PATH: str = "./phi-3-mini-4k-instruct-q4_k_m.gguf" # Path to the GGUF LLM model file
CHROMA_PERSIST_DIR: str = "./chroma_db" # Directory to store the vector database
CHROMA_COLLECTION_NAME: str = "pdf_chunks" # Name for the ChromaDB collection

# --- Embedding Model ---
EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2' # Sentence Transformer model for embeddings

# --- Text Chunking ---
# Note: Chunking logic might be based on headings now, these might be less relevant
# CHUNK_SIZE: int = 400 # Target character size for text chunks (if using size-based chunking)
# CHUNK_OVERLAP: int = 50 # Character overlap between chunks (if using size-based chunking)

# --- Retrieval ---
N_RETRIEVAL_RESULTS: int = 3 # Number of relevant chunks to retrieve
DISTANCE_THRESHOLD: float = 1.5 # Max distance threshold for relevance (needs tuning, lower is stricter)

# --- LLM Loading ---
LLM_N_CTX: int = 4096 # Context window size for the LLM
LLM_N_THREADS: int = 8  # Number of CPU threads for LLM inference (adjust to your system)
LLM_N_GPU_LAYERS: int = 0 # Number of LLM layers to offload to GPU (0 for CPU only)

# --- LLM Generation ---
LLM_MAX_NEW_TOKENS: int = 512 # Maximum number of tokens for the LLM to generate
LLM_TEMPERATURE: float = 0.5 # Controls randomness (lower for factual, higher for creative)
LLM_STOP_TOKENS: list[str] = ['<|end|>', '<|endoftext|>'] # Tokens that signal the LLM to stop generating

# --- Logging ---
LOG_LEVEL: str = "INFO" # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Log message format