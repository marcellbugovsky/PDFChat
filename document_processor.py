# document_processor.py

import pypdf
import logging
import re
import sys
from typing import List, Tuple

# Import configuration constants
from config import PDF_FILE_PATH, LOG_LEVEL, LOG_FORMAT

# Configure logging for this module
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Headlines specific to the CV structure used for chunking
# Keep these in German as they need to match the text in the German CV
CV_HEADLINES = [
    "BERUFLICHER WERDEGANG",
    "AUSBILDUNG",
    "BESONDERE KENNTNISSE",
    "HOBBYS & INTERESSEN",
    "PERSÖNLICHES"
]

def load_and_clean_pdf_text(pdf_path: str = PDF_FILE_PATH) -> str:
    """
    Loads text content from a PDF file and performs basic cleaning.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A single string containing the cleaned text from the PDF, or empty string on error.
    """
    logger.info(f"Loading PDF from path: {pdf_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        logger.info(f"Found {len(reader.pages)} page(s).")
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Replace common ligatures and normalize whitespace
                    page_text = page_text.replace("ﬁ", "fi").replace("ﬀ", "ff")
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    full_text += page_text + " " # Add space between page contents
                else:
                    logger.warning(f"No text extracted from page {page_num + 1}.")
            except Exception as e:
                # Log error for specific page but continue processing others
                logger.error(f"Error extracting text from page {page_num + 1}: {e}")

        if not full_text:
             logger.error(f"No text could be extracted from the PDF: {pdf_path}")
             return ""

        logger.info(f"Successfully loaded and extracted text from {len(reader.pages)} pages.")
        return full_text.strip() # Final trim

    except FileNotFoundError:
        logger.error(f"PDF file not found at the specified path: {pdf_path}")
        return ""
    except pypdf.errors.PdfReadError as e:
         logger.error(f"Error reading PDF file (possibly corrupted or password protected): {pdf_path} - {e}")
         return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the PDF: {e}", exc_info=True)
        return ""

def chunk_by_headings(full_text: str) -> List[str]:
    """
    Splits the input text into chunks based on predefined German CV headings.

    Args:
        full_text: The entire text content extracted from the PDF.

    Returns:
        A list of strings, where each string is a chunk corresponding to a CV section.
    """
    if not full_text:
        logger.warning("Input text for chunking is empty.")
        return []

    logger.info("Chunking text based on CV headings...")
    chunks: List[str] = []
    last_index: int = 0

    # --- Find start indices of all known headings ---
    heading_indices: List[Tuple[int, str]] = []
    for headline in CV_HEADLINES:
        try:
            # Case-insensitive search for whole word headings
            # Using finditer to get all occurrences if needed, but take first for now
            matches = list(re.finditer(r'\b' + re.escape(headline) + r'\b', full_text, re.IGNORECASE))
            if matches:
                 # Find the first occurrence of this headline
                 start_index = matches[0].start()
                 heading_indices.append((start_index, headline))
            else:
                 logger.debug(f"Heading '{headline}' not found in document.")
        except re.error as e:
             logger.warning(f"Regex error searching for heading '{headline}': {e}")

    # Sort headings by their appearance order in the text
    heading_indices.sort(key=lambda x: x[0])

    # --- Create Chunks ---
    # 1. Handle text before the first heading (Introduction)
    if heading_indices and heading_indices[0][0] > 0:
        intro_chunk_text = full_text[:heading_indices[0][0]].strip()
        if intro_chunk_text:
            # Prepend a generic title for clarity
            chunks.append(f"Introduction:\n{intro_chunk_text}")
            logger.debug(f"Created Introduction chunk (length: {len(intro_chunk_text)}).")
            last_index = heading_indices[0][0]
    elif not heading_indices:
        logger.warning("No defined headings found. Returning the entire text as one chunk.")
        return [full_text] # Return full text if no headings are found

    # 2. Create chunks for each section between headings
    for i, (start_index, headline) in enumerate(heading_indices):
         # Determine the end index for the current chunk
         if i + 1 < len(heading_indices):
             # End before the next heading starts
             end_index = heading_indices[i+1][0]
         else:
             # This is the last heading, go to the end of the text
             end_index = len(full_text)

         # Extract the chunk content (includes the headline itself)
         chunk_content = full_text[start_index:end_index].strip()

         if chunk_content:
             chunks.append(chunk_content)
             logger.debug(f"Created chunk for '{headline}' (length: {len(chunk_content)}).")
         else:
             logger.warning(f"Empty chunk generated for heading '{headline}', skipping.")

    logger.info(f"Successfully created {len(chunks)} chunks based on headings.")
    return chunks