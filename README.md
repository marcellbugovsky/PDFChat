# PDFChat

Mit diesem Python-Projekt können Sie lokal mit Ihren PDF-Dokumenten chatten! Es verwendet Retrieval-Augmented Generation (RAG) mit Open-Source Embedding-Modellen und Large Language Models (LLMs), um Fragen ausschließlich basierend auf dem Inhalt der bereitgestellten PDF-Datei zu beantworten. Alle Modelle und Daten bleiben lokal auf Ihrem System.

## Projektbeschreibung

PDFChat ermöglicht es Benutzern, Fragen zu einem bestimmten PDF-Dokument zu stellen. Das System extrahiert den Text aus dem PDF, teilt ihn in sinnvolle Abschnitte auf, erstellt semantische Vektoren (Embeddings) dieser Abschnitte und speichert sie in einer lokalen Vektordatenbank (ChromaDB). Wenn eine Frage gestellt wird, sucht das System nach den relevantesten Textabschnitten in der Datenbank und übergibt diese zusammen mit der Frage an ein lokales Large Language Model (LLM), das mit `llama-cpp-python` ausgeführt wird. Das LLM generiert dann eine Antwort, die strikt auf den bereitgestellten Informationen aus dem PDF basiert.

**Wichtiger Hinweis:** Die aktuelle Implementierung der Textextraktion und -segmentierung (`document_processor.py`) ist speziell auf die Struktur eines bestimmten deutschen Lebenslaufs (basierend auf Überschriften wie "BERUFLICHER WERDEGANG", "AUSBILDUNG" etc.) zugeschnitten. Für andere PDF-Dokumente müsste die Funktion `chunk_by_headings` angepasst oder durch eine allgemeinere Chunking-Strategie (z. B. feste Größe mit Überlappung) ersetzt werden.

## Features

* **Lokale Verarbeitung:** Alle Komponenten (Embedding-Modell, LLM, Vektordatenbank) laufen lokal, es werden keine Daten an externe APIs gesendet.
* **RAG-Pipeline:** Implementiert eine vollständige Retrieval-Augmented Generation Pipeline.
* **PDF-Verarbeitung:** Extrahiert Text aus PDF-Dateien (`pypdf`) und segmentiert ihn (aktuell CV-spezifisch).
* **Vektor-Speicherung:** Nutzt ChromaDB zur lokalen Speicherung und Abfrage von Text-Embeddings.
* **Flexible Modelle:** Verwendet Sentence Transformers für Embeddings und ein GGUF-Modell über `llama-cpp-python` für die Sprachgenerierung. Modelle und Parameter sind über `config.py` konfigurierbar.
* **Zwei Schnittstellen:**
    * **Streamlit Web UI (`app.py`):** Eine benutzerfreundliche Web-Oberfläche für den Chat.
    * **Kommandozeilen-Interface (`main.py`):** Eine alternative Möglichkeit zur Interaktion über die Konsole.

## Funktionsweise (RAG)

1.  **Ingestion:** Das PDF wird geladen, der Text extrahiert und in Abschnitte (Chunks) unterteilt.
2.  **Embedding & Indexing:** Für jeden Chunk wird ein Embedding-Vektor mittels Sentence Transformers erstellt und zusammen mit dem Text in der ChromaDB gespeichert. (Dies geschieht automatisch beim ersten Start, wenn die Datenbank leer ist).
3.  **Retrieval:** Bei einer Benutzeranfrage wird diese ebenfalls in einen Embedding-Vektor umgewandelt. Die Vektordatenbank wird nach den Chunks durchsucht, deren Embeddings diesem Anfrage-Vektor am ähnlichsten sind (basierend auf der euklidischen Distanz).
4.  **Generation:** Die relevantesten Chunks werden als Kontext zusammen mit der ursprünglichen Anfrage an das LLM übergeben. Das LLM wird angewiesen, die Frage ausschließlich basierend auf diesem Kontext zu beantworten.

## Verwendete Technologien

* **Sprache:** Python 3
* **Kernbibliotheken:**
    * `streamlit`: Für die Web-Benutzeroberfläche.
    * `pypdf`: Zum Lesen und Extrahieren von Text aus PDF-Dateien.
    * `sentence-transformers`: Zum Erstellen von Text-Embeddings.
    * `chromadb`: Als lokale Vektordatenbank.
    * `llama-cpp-python`: Zum Laden und Ausführen von GGUF LLM-Modellen lokal.
    * `torch`: Als Backend für Sentence Transformers (und potenziell llama-cpp-python).
    * `transformers`, `accelerate`: Oft Abhängigkeiten für `sentence-transformers` oder `llama-cpp-python`.

## Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/marcellbugovsky/PDFChat.git](https://github.com/marcellbugovsky/PDFChat.git)
    cd PDFChat
    ```
2.  **Modell und PDF platzieren:**
    * Lade das gewünschte GGUF LLM-Modell herunter (z.B. `phi-3-mini-4k-instruct-q4_k_m.gguf` von Hugging Face) und platziere es im Hauptverzeichnis oder passe den Pfad in `config.py` (`LLM_MODEL_PATH`) an.
    * Platziere die PDF-Datei, die analysiert werden soll, als `document.pdf` im Hauptverzeichnis oder passe den Pfad in `config.py` (`PDF_FILE_PATH`) an.

3.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    venv\Scripts\activate    # Windows
    ```
4.  **Abhängigkeiten installieren:**
    * **Wichtig für `llama-cpp-python`:** Die Installation kompiliert Code und benötigt möglicherweise C++ Build-Tools (wie `build-essential` unter Linux oder Visual Studio Build Tools unter Windows). Befolge die Installationsanweisungen von `llama-cpp-python` für dein System (ggf. mit Hardwarebeschleunigung wie CUDA/Metal).
    ```bash
    pip install -r requirements.txt
    ```

## Verwendung

1.  **Indexierung (Automatisch):** Beim ersten Start von `app.py` oder `main.py` wird automatisch geprüft, ob die Vektordatenbank für die konfigurierte PDF-Datei existiert und gefüllt ist. Wenn nicht, wird das PDF verarbeitet und indexiert. Dies kann einen Moment dauern.

2.  **Option A: Streamlit Web UI (Empfohlen):**
    * Starte die Streamlit-Applikation:
      ```bash
      streamlit run app.py
      ```
    * Öffne deinen Webbrowser unter der angezeigten lokalen Adresse (normalerweise `http://localhost:8501`).
    * Stelle deine Fragen im Chat-Interface.

3.  **Option B: Kommandozeilen-Interface:**
    * Starte das CLI-Skript:
      ```bash
      python main.py
      ```
    * Folge den Anweisungen im Terminal, um Fragen zu stellen. Tippe `quit` oder `exit`, um das Programm zu beenden.

## Konfiguration (`config.py`)

Die Datei `config.py` enthält wichtige Einstellungen:

* **Pfade:** Speicherorte für die PDF-Datei, das LLM-Modell und die ChromaDB-Datenbank.
* **Modelle:** Name des Sentence Transformer Modells für Embeddings.
* **Retrieval:** Anzahl der abzurufenden Chunks (`N_RETRIEVAL_RESULTS`) und der Schwellenwert für die Relevanz (`DISTANCE_THRESHOLD`).
* **LLM-Parameter:** Kontextfenstergröße (`LLM_N_CTX`), CPU-Threads (`LLM_N_THREADS`), GPU-Layer (`LLM_N_GPU_LAYERS`), maximale Token-Generierung (`LLM_MAX_NEW_TOKENS`), Temperatur (`LLM_TEMPERATURE`), Stop-Token (`LLM_STOP_TOKENS`).
* **Logging:** Log-Level und Format.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.