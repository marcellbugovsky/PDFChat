# llm_handler.py

from llama_cpp import Llama
import logging
# Importiere Konstanten aus der Konfigurationsdatei
from config import LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_N_GPU_LAYERS, \
                   LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_STOP_TOKENS, \
                   LOG_LEVEL, LOG_FORMAT

# Logging Konfiguration (falls hier noch nicht global gesetzt)
# Es ist oft besser, Logging im Hauptskript zu konfigurieren
# logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__) # Verwende einen spezifischen Logger für das Modul

class LLMHandler:
    """
    Kapselt das Laden und die Interaktion mit dem lokalen LLM via llama-cpp-python.
    """
    def __init__(self):
        self.model_path = LLM_MODEL_PATH
        self.n_ctx = LLM_N_CTX
        self.n_threads = LLM_N_THREADS
        self.n_gpu_layers = LLM_N_GPU_LAYERS
        self.llm = self._load_llm()

    def _load_llm(self) -> Llama | None:
        """Lädt das GGUF-Modell mit llama-cpp-python."""
        try:
            logger.info(f"Lade LLM via llama-cpp-python von: '{self.model_path}'...")
            llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False # Reduziert die ausführlichen Logs von llama.cpp selbst
            )
            logger.info("LLM erfolgreich geladen (llama-cpp-python).")
            return llm
        except Exception as e:
            # Spezifischere Fehlermeldung für Installationsprobleme
            if "DLL load failed" in str(e) or "Can't find llama backend" in str(e):
                 logger.error("Konnte llama.cpp Backend nicht laden. Ist llama-cpp-python korrekt installiert (benötigt C++ Compiler)?")
            logger.error(f"Fataler Fehler beim Laden des LLM: {e}", exc_info=True)
            # Statt raise hier None zurückgeben, damit das Hauptprogramm den Fehler behandeln kann
            # raise # Kritischer Fehler, wenn das Laden fehlschlägt
            return None

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Generiert eine Antwort mittels LLM basierend auf dem Kontext."""
        if not self.llm:
            logger.error("LLM ist nicht geladen und kann keine Antwort generieren.")
            return "Fehler: LLM nicht verfügbar."
        if not context_chunks:
            # Diese Prüfung könnte auch im Hauptskript erfolgen
            logger.warning("Keine Kontext-Chunks für die Generierung erhalten.")
            return "Ich konnte keine relevanten Informationen im Dokument finden, um diese Frage zu beantworten."

        context_str = "\n\n---\n\n".join(context_chunks)
        logger.debug(f"Kontext für LLM (gekürzt): {context_str[:500]}...") # DEBUG Level für vollen Kontext

        # Prompt-Vorlage definieren (Deutsch)
        # Sicherstellen, dass die Formatierung klar ist
        prompt_template = f"""Anweisung: Beantworte die folgende Frage ausschließlich basierend auf dem bereitgestellten Kontext. Wenn die Antwort nicht im Kontext enthalten ist, schreibe "Die Antwort ist im vorliegenden Dokument nicht enthalten.". Antworte auf Deutsch.

Kontext:
{context_str}

Frage: {query}

Antwort:""" # Leerzeichen am Ende entfernt, um sofortige Stops zu vermeiden

        logger.info("Sende Prompt an LLM (llama-cpp)...")
        try:
            response_object = self.llm.create_completion(
                prompt=prompt_template,
                max_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                stop=['<|end|>'] # << ANGEPASST: Nur noch das primäre Stop-Token verwenden
                # stop=LLM_STOP_TOKENS # Alte Version mit mehr Tokens
            )
            # Antwort extrahieren und bereinigen
            answer = response_object['choices'][0]['text'].strip()
            logger.info("LLM Antwort erhalten (llama-cpp).")
            # Zusätzliche Bereinigung: Manchmal fügen Modelle den Stop-Token selbst hinzu
            if answer.endswith('<|end|>'):
                 answer = answer[:-len('<|end|>')].strip()

            # Fallback, falls die Antwort leer ist nach dem Strippen
            if not answer:
                 logger.warning("LLM hat eine leere Antwort generiert.")
                 # Eventuell hier die Standard-Antwort "nicht gefunden" zurückgeben?
                 # return "Das Modell hat keine spezifische Antwort generiert."
                 return "Die Antwort ist im vorliegenden Dokument nicht enthalten." # Konsistent mit Anweisung

            return answer
        except Exception as e:
            logger.error(f"Fehler bei der LLM-Generierung (llama-cpp): {e}", exc_info=True)
            return "Es gab einen Fehler bei der Generierung der Antwort."