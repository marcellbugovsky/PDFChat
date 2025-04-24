# main.py

import logging
import sys
from config import LOG_LEVEL, LOG_FORMAT, N_RETRIEVAL_RESULTS, DISTANCE_THRESHOLD, \
                   CHUNK_SIZE, CHUNK_OVERLAP, PDF_FILE_PATH
from document_processor import load_and_clean_pdf_text, chunk_by_headlines
from vector_store_manager import VectorStoreManager
from llm_handler import LLMHandler

# Logging Konfiguration
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

def run_chatbot():
    """Initialisiert und startet den Chatbot."""
    try:
        # --- Initialisierung ---
        logging.info("Initialisiere Komponenten...")
        vector_manager = VectorStoreManager()
        llm_handler = LLMHandler()
        logging.info("Komponenten initialisiert.")

        # --- Sicherstellen, dass Vektor-Store bereit ist ---
        if vector_manager.needs_indexing():
            logging.info("Vektor-Store muss indiziert werden.")
            pdf_text = load_and_clean_pdf_text() # Nutzt Pfad aus config
            if not pdf_text:
                 logging.error("Konnte PDF nicht laden, Abbruch.")
                 return # Beendet die Funktion, wenn PDF nicht geladen werden kann

            text_chunks = chunk_by_headlines(pdf_text)
            if not text_chunks:
                 logging.error("Konnte keine Chunks erstellen, Abbruch.")
                 return

            success = vector_manager.index_documents(text_chunks)
            if not success:
                 logging.error("Indizierung fehlgeschlagen, Abbruch.")
                 return
        else:
            logging.info("Nutze existierenden Vektor-Store.")

        # --- Interaktive Schleife ---
        print("\nPDF Chatbot Initialisiert. Fragen zum Dokument stellen.")
        print("Geben Sie 'quit' oder 'exit' ein, um zu beenden.")

        while True:
            user_query = input("\nIhre Frage: ")
            if user_query.lower() in ['quit', 'exit']:
                print("Beende Chatbot.")
                break
            if not user_query.strip():
                print("Bitte geben Sie eine Frage ein.")
                continue

            # 1. Chunks abrufen
            relevant_docs, distances = vector_manager.retrieve_relevant_chunks(
                user_query, N_RETRIEVAL_RESULTS
            )

            # 2. Guardrail & Antwortgenerierung
            answer = ""
            if relevant_docs and distances[0] <= DISTANCE_THRESHOLD:
                logging.info(f"Relevante Chunks gefunden (beste Distanz: {distances[0]:.4f}). Generiere Antwort...")
                answer = llm_handler.generate_answer(user_query, relevant_docs)
            elif relevant_docs:
                logging.warning(f"Chunks gefunden, aber beste Distanz ({distances[0]:.4f}) > Schwellenwert ({DISTANCE_THRESHOLD}).")
                answer = "Ich habe verwandte Informationen gefunden, aber sie scheinen nicht relevant genug zu sein, um Ihre Frage sicher zu beantworten."
            else:
                logging.info("Keine relevanten Chunks gefunden.")
                answer = "Ich konnte keine relevanten Informationen im Dokument finden, um diese Frage zu beantworten."

            # 3. Antwort ausgeben
            print(f"\nAntwort:\n{answer}")

    except Exception as e:
        logging.error(f"Ein unerwarteter Fehler ist im Hauptablauf aufgetreten: {e}", exc_info=True)
        print("Ein kritischer Fehler ist aufgetreten. Bitte überprüfen Sie die Logs.")

if __name__ == "__main__":
    run_chatbot()