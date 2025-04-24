# vector_store_manager.py

import chromadb
import logging
import os
import sys
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Import configuration constants
from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME,
    LOG_LEVEL, LOG_FORMAT
)

# Configure logging for this module
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Handles creation, population, and querying of the ChromaDB vector store."""

    def __init__(self):
        """Initializes embedding model and ChromaDB client/collection."""
        self.persist_directory: str = CHROMA_PERSIST_DIR
        self.collection_name: str = CHROMA_COLLECTION_NAME
        self.embedding_model_name: str = EMBEDDING_MODEL_NAME
        self.embedding_model: Optional[SentenceTransformer] = self._load_embedding_model()
        self.client: Optional[chromadb.PersistentClient] = self._initialize_client()
        self.collection: Optional[chromadb.Collection] = self._get_or_create_collection()

        if not self.embedding_model or not self.client or not self.collection:
             raise RuntimeError("Failed to initialize VectorStoreManager components.")

    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        """Loads the Sentence Transformer embedding model."""
        try:
            logger.info(f"Loading embedding model: '{self.embedding_model_name}'...")
            # Consider adding device='cpu' if GPU issues arise or explicit CPU is needed
            model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}", exc_info=True)
            return None # Failure indication

    def _initialize_client(self) -> Optional[chromadb.PersistentClient]:
        """Initializes the persistent ChromaDB client."""
        try:
            logger.info(f"Initializing ChromaDB client at '{self.persist_directory}'")
            os.makedirs(self.persist_directory, exist_ok=True) # Ensure directory exists
            client = chromadb.PersistentClient(path=self.persist_directory)
            return client
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}", exc_info=True)
            return None # Failure indication

    def _get_or_create_collection(self) -> Optional[chromadb.Collection]:
        """Gets or creates the specified ChromaDB collection."""
        if not self.client:
            logger.error("ChromaDB client not initialized.")
            return None
        try:
            logger.info(f"Getting or creating collection: '{self.collection_name}'")
            # get_or_create handles both cases
            collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' is ready. Current item count: {collection.count()}")
            return collection
        except Exception as e:
            logger.error(f"Error getting or creating collection '{self.collection_name}': {e}", exc_info=True)
            return None # Failure indication

    def needs_indexing(self) -> bool:
        """Checks if the collection is empty and likely requires indexing."""
        if self.collection:
            try:
                return self.collection.count() == 0
            except Exception as e:
                logger.error(f"Could not get count for collection '{self.collection_name}': {e}")
                return True # Assume indexing needed if count fails
        return True # Assume indexing needed if collection doesn't exist

    def index_documents(self, chunks: List[str]) -> bool:
        """
        Indexes the provided text chunks into the ChromaDB collection.
        Clears the existing collection before adding new data.

        Args:
            chunks: A list of text strings to index.

        Returns:
            True if indexing was successful, False otherwise.
        """
        if not self.collection or not self.embedding_model:
            logger.error("Cannot index: Vector store or embedding model not initialized.")
            return False
        if not chunks:
            logger.warning("No text chunks provided for indexing.")
            return False

        try:
             # --- Clear existing collection before indexing ---
             # This ensures fresh indexing. Alternatives exist for updating.
             count = self.collection.count()
             if count > 0:
                 logger.warning(f"Clearing existing collection '{self.collection_name}' ({count} items) before re-indexing.")
                 self.client.delete_collection(name=self.collection_name)
                 self.collection = self.client.create_collection(name=self.collection_name)
             # ------------------------------------------------

             logger.info(f"Generating embeddings for {len(chunks)} chunks...")
             # Encode chunks using the loaded sentence transformer model
             embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
             logger.info("Embeddings generated.")

             # Prepare data for ChromaDB: IDs should be unique strings
             chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

             logger.info(f"Adding {len(chunks)} chunks to ChromaDB collection '{self.collection_name}'...")
             # Add documents, embeddings, and IDs to the collection
             self.collection.add(
                 embeddings=embeddings.tolist(), # ChromaDB expects lists of floats
                 documents=chunks,
                 ids=chunk_ids
             )
             logger.info(f"Chunks added to ChromaDB successfully. New count: {self.collection.count()}")
             return True # Success
        except Exception as e:
             logger.error(f"Error indexing documents: {e}", exc_info=True)
             return False # Failure

    def retrieve_relevant_chunks(self, query: str, n_results: int) -> Tuple[List[str], List[float]]:
        """
        Retrieves the most relevant text chunks for a given query from ChromaDB.

        Args:
            query: The user's query string.
            n_results: The maximum number of chunks to retrieve.

        Returns:
            A tuple containing:
                - A list of the retrieved document text chunks.
                - A list of the corresponding distances (lower is more similar).
        """
        if not query or not self.collection or not self.embedding_model:
            logger.warning("Cannot retrieve: Missing query, collection, or embedding model.")
            return [], []
        try:
            logger.info(f"Generating embedding for query: '{query[:50]}...'")
            query_embedding = self.embedding_model.encode([query]) # Encode query
            logger.info("Query embedding generated.")

            logger.info(f"Querying ChromaDB collection '{self.collection_name}' for {n_results} results...")
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(), # Query with the embedding
                n_results=n_results,
                include=['documents', 'distances'] # Request documents and distances
            )
            logger.info("ChromaDB query complete.")

            # Safely extract results, handling potential missing keys or empty lists
            retrieved_docs = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]

            # Ensure proper pairing and filter potential None values if API behaved unexpectedly
            valid_results = [
                (doc, dist) for doc, dist in zip(retrieved_docs, distances)
                if doc is not None and dist is not None
            ]
            retrieved_docs = [doc for doc, dist in valid_results]
            distances = [dist for doc, dist in valid_results]

            logger.info(f"Retrieved {len(retrieved_docs)} chunks.")
            return retrieved_docs, distances
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return [], [] # Return empty lists on error