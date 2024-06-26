# keyword_extraction.py
import spacy
import logging
from typing import List

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords from the given text using spaCy.

    Args:
        text (str): The text to extract keywords from.
        num_keywords (int): The number of keywords to extract.

    Returns:
        List[str]: The list of extracted keywords.
    """
    try:
        # Process the text with spaCy
        doc = nlp(text)

        # Extract keywords using spaCy's built-in functionality
        keywords = [chunk.text.lower() for chunk in doc.noun_chunks if chunk.root.pos_ in ["NOUN", "PROPN"]]

        # Remove duplicates and sort by frequency
        keyword_counts = {keyword: keywords.count(keyword) for keyword in set(keywords)}
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

        # Return the top N keywords
        return [keyword for keyword, _ in sorted_keywords[:num_keywords]]
    except Exception as e:
        logging.error(f"Error in extract_keywords: {str(e)}")
        return []