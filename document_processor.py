# document_processor.py

import pypdf
import logging
import re
import sys
from config import PDF_FILE_PATH, LOG_LEVEL, LOG_FORMAT # Importiere Konfiguration

# Logging Konfiguration
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Bekannte Überschriften aus dem Lebenslauf (Großschreibung beachten!)
# Fügen Sie hier ggf. weitere hinzu oder passen Sie sie an.
# Die Reihenfolge ist wichtig für die korrekte Aufteilung.
HEADLINES = [
    "BERUFLICHER WERDEGANG",
    "AUSBILDUNG",
    "BESONDERE KENNTNISSE",
    "HOBBYS & INTERESSEN",
    "PERSÖNLICHES"
]

def load_and_clean_pdf_text(pdf_path: str = PDF_FILE_PATH) -> str:
    """Lädt Text aus einer PDF-Datei und führt grundlegende Bereinigungen durch."""
    logging.info(f"Lade PDF von: {pdf_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Ersetze häufige Ligaturen und normalisiere Whitespace direkt hier
                    page_text = page_text.replace("ﬁ", "fi").replace("ﬀ", "ff") # Beispiel für Ligaturen
                    page_text = re.sub(r'\s+', ' ', page_text) # Ersetze mehrere Leerzeichen/Umbrüche durch eines
                    full_text += page_text.strip() + " " # Füge Leerzeichen statt Newline hinzu
                else:
                    logging.warning(f"Konnte keinen Text von Seite {page_num + 1} extrahieren.")
            except Exception as e:
                logging.error(f"Fehler beim Extrahieren von Seite {page_num + 1}: {e}")

        if not full_text:
             logging.error(f"Kein Text konnte aus {pdf_path} extrahiert werden.")
             return ""

        logging.info(f"Erfolgreich {len(reader.pages)} Seiten geladen und Text extrahiert.")
        # Finale Bereinigung des gesamten Textes
        full_text = full_text.strip()
        return full_text
    except FileNotFoundError:
        logging.error(f"Fehler: PDF-Datei nicht gefunden unter {pdf_path}")
        return ""
    except Exception as e:
        logging.error(f"Fehler beim Laden der PDF-Datei: {e}", exc_info=True)
        return ""

def chunk_by_headlines(full_text: str) -> list[str]:
    """Teilt den Text basierend auf einer vordefinierten Liste von Überschriften."""
    if not full_text:
        return []

    logging.info("Teile Text nach Überschriften...")
    chunks = []
    last_index = 0

    # Finde die Startposition der ersten bekannten Überschrift
    first_headline_index = -1
    first_headline = ""
    for headline in HEADLINES:
        try:
            # Suche nach der Überschrift, ignoriere Groß-/Kleinschreibung und umgebende Leerzeichen
            # Verwende einen regulären Ausdruck für mehr Flexibilität
            match = re.search(r'\b' + re.escape(headline) + r'\b', full_text, re.IGNORECASE)
            if match:
                current_index = match.start()
                if first_headline_index == -1 or current_index < first_headline_index:
                    first_headline_index = current_index
                    first_headline = headline # Behalte die Original-Überschrift für den Chunk
        except re.error as e:
             logging.warning(f"Fehler bei der Regex-Suche nach '{headline}': {e}")
             continue # Ignoriere Fehler und mache weiter


    # Füge den Text vor der ersten Überschrift als ersten Chunk hinzu (Einleitungsteil)
    if first_headline_index > 0:
        intro_chunk = full_text[:first_headline_index].strip()
        if intro_chunk: # Nur hinzufügen, wenn nicht leer
             # Optional: Füge eine generische Überschrift hinzu
             chunks.append(f"Einleitung:\n{intro_chunk}")
             last_index = first_headline_index
             logging.info(f"Einleitungs-Chunk erstellt (Länge: {len(intro_chunk)}).")
    elif first_headline_index == 0:
         last_index = 0 # Erste Überschrift ist ganz am Anfang
    else:
         logging.warning("Keine der definierten Überschriften im Text gefunden. Gebe gesamten Text als einen Chunk zurück.")
         return [full_text] # Keine Überschriften gefunden, ganzer Text ist ein Chunk


    # Iteriere durch die definierten Überschriften, um die Abschnitte zu finden
    for i, headline in enumerate(HEADLINES):
         try:
             # Finde den Startindex der aktuellen Überschrift *nach* dem letzten Index
             # Wieder mit Regex für Flexibilität
             match = re.search(r'\b' + re.escape(headline) + r'\b', full_text[last_index:], re.IGNORECASE)
             if not match:
                 logging.debug(f"Überschrift '{headline}' nicht nach Index {last_index} gefunden.")
                 continue # Überschrift nicht gefunden in diesem Teil, überspringen

             start_index = last_index + match.start()

             # Finde den Startindex der *nächsten* Überschrift in der Liste
             next_headline_index = len(full_text) # Standardmäßig bis zum Ende des Textes
             for next_headline in HEADLINES[i+1:]:
                  try:
                      next_match = re.search(r'\b' + re.escape(next_headline) + r'\b', full_text[start_index + len(headline):], re.IGNORECASE)
                      if next_match:
                           # Berechne den absoluten Index der nächsten Überschrift
                           potential_next_index = start_index + len(headline) + next_match.start()
                           # Nehme den frühesten gefundenen nächsten Index
                           next_headline_index = min(next_headline_index, potential_next_index)
                  except re.error as e:
                       logging.warning(f"Fehler bei der Regex-Suche nach nächster Überschrift '{next_headline}': {e}")


             # Extrahiere den Textblock für die aktuelle Überschrift
             # Der Chunk beginnt mit der Überschrift selbst
             chunk_text_content = full_text[start_index:next_headline_index].strip()

             if chunk_text_content: # Nur hinzufügen, wenn der Chunk Inhalt hat
                 chunks.append(chunk_text_content)
                 logging.info(f"Chunk für '{headline}' erstellt (Länge: {len(chunk_text_content)}).")
                 last_index = next_headline_index # Aktualisiere für die nächste Suche
             else:
                  logging.warning(f"Leerer Chunk für Überschrift '{headline}' gefunden, wird übersprungen.")
                  # Optional: Aktualisiere last_index trotzdem, um Sprünge zu vermeiden
                  # last_index = next_headline_index

         except re.error as e:
              logging.warning(f"Fehler bei der Regex-Suche nach '{headline}': {e}")
              continue # Mache mit der nächsten Überschrift weiter

    # Fallback: Füge den Rest des Textes hinzu, falls nach der letzten Überschrift noch etwas kommt
    # This might not be needed if the logic above correctly finds the end
    # if last_index < len(full_text):
    #    remaining_chunk = full_text[last_index:].strip()
    #    if remaining_chunk:
    #        chunks.append(f"Sonstiges:\n{remaining_chunk}") # Oder ohne Überschrift
    #        logging.info(f"Restlicher Text als letzter Chunk hinzugefügt (Länge: {len(remaining_chunk)}).")


    logging.info(f"Insgesamt {len(chunks)} Chunks nach Überschriften erstellt.")
    return chunks


# --- Alte chunk_text Funktion (kann entfernt oder auskommentiert werden) ---
# def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 25) -> list[str]:
#     # ... (alter Code)
#     pass