# vector_store_manager.py

import chromadb
import logging
import os
from sentence_transformers import SentenceTransformer
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME

logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    def __init__(self):
        self.persist_directory = CHROMA_PERSIST_DIR
        self.collection_name = CHROMA_COLLECTION_NAME
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.embedding_model = self._load_embedding_model()
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()

    def _load_embedding_model(self):
        try:
            logging.info(f"Lade Embedding-Modell: '{self.embedding_model_name}'...")
            model = SentenceTransformer(self.embedding_model_name)
            logging.info("Embedding-Modell erfolgreich geladen.")
            return model
        except Exception as e:
            logging.error(f"Fehler beim Laden des Embedding-Modells: {e}", exc_info=True)
            raise # Fehler weitergeben, da kritisch

    def _initialize_client(self):
        try:
            logging.info(f"Initialisiere ChromaDB Client unter '{self.persist_directory}'")
            os.makedirs(self.persist_directory, exist_ok=True)
            return chromadb.PersistentClient(path=self.persist_directory)
        except Exception as e:
            logging.error(f"Fehler beim Initialisieren des ChromaDB Clients: {e}", exc_info=True)
            raise

    def _get_or_create_collection(self):
        try:
            logging.info(f"Hole oder erstelle Collection: '{self.collection_name}'")
            collection = self.client.get_or_create_collection(name=self.collection_name)
            logging.info(f"Collection '{self.collection_name}' bereit. Aktuelle Anzahl Items: {collection.count()}")
            return collection
        except Exception as e:
            logging.error(f"Fehler beim Holen/Erstellen der Collection: {e}", exc_info=True)
            return None # Oder Fehler weitergeben

    def needs_indexing(self) -> bool:
        """Prüft, ob die Collection leer ist und Indizierung benötigt."""
        if self.collection:
            return self.collection.count() == 0
        return True # Wenn keine Collection vorhanden ist, muss indiziert werden

    def index_documents(self, chunks: list[str]):
        """Indiziert die übergebenen Text-Chunks (Embeddings + Speichern)."""
        if not self.collection or not self.embedding_model:
            logging.error("Vector Store oder Embedding Modell nicht initialisiert.")
            return False
        if not chunks:
            logging.warning("Keine Chunks zum Indizieren übergeben.")
            return False

        try:
             # Collection leeren, wenn sie schon existiert und neu indiziert werden soll
             # (Alternative: update statt add, oder komplexere Logik für geänderte Chunks)
             count = self.collection.count()
             if count > 0:
                 logging.warning(f"Bestehende Collection '{self.collection_name}' mit {count} Items wird für Neuindizierung gelöscht.")
                 self.client.delete_collection(name=self.collection_name)
                 self.collection = self.client.create_collection(name=self.collection_name)

             logging.info(f"Generiere Embeddings für {len(chunks)} Chunks...")
             embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
             logging.info("Embeddings generiert.")

             chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

             logging.info(f"Füge {len(chunks)} Chunks zur ChromaDB Collection hinzu...")
             self.collection.add(
                 embeddings=embeddings.tolist(),
                 documents=chunks,
                 ids=chunk_ids
             )
             logging.info(f"Chunks erfolgreich zu ChromaDB hinzugefügt. Neue Anzahl: {self.collection.count()}")
             return True
        except Exception as e:
             logging.error(f"Fehler beim Indizieren der Dokumente: {e}", exc_info=True)
             return False

    def retrieve_relevant_chunks(self, query: str, n_results: int) -> tuple[list[str], list[float]]:
        """Ruft die relevantesten Chunks für eine Anfrage ab."""
        if not query or not self.collection or not self.embedding_model:
            return [], []
        try:
            logging.info(f"Generiere Embedding für Anfrage: '{query[:50]}...'")
            query_embedding = self.embedding_model.encode([query])
            logging.info("Anfrage-Embedding generiert.")

            logging.info(f"Suche in ChromaDB nach {n_results} relevantesten Chunks...")
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'distances']
            )
            logging.info("ChromaDB-Abfrage abgeschlossen.")

            retrieved_docs = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]

            valid_results = [(doc, dist) for doc, dist in zip(retrieved_docs, distances) if doc is not None and dist is not None]
            retrieved_docs = [doc for doc, dist in valid_results]
            distances = [dist for doc, dist in valid_results]

            return retrieved_docs, distances
        except Exception as e:
            logging.error(f"Fehler beim Abrufen der Chunks: {e}", exc_info=True)
            return [], []