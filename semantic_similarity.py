# semantic_similarity.py
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load the SentenceTransformer model
sentence_transformer = SentenceTransformer('all-mpnet-base-v2')

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by tokenizing, lowercasing, removing stop words and punctuation.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    # Tokenize and lowercase the text
    doc = nlp(text.lower())
    # Remove stop words and punctuation
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate the semantic similarity between two texts using SentenceTransformer.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The semantic similarity score.
    """
    try:
        # Preprocess the text
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)
        # Generate sentence embeddings using SentenceTransformer
        embeddings1 = sentence_transformer.encode([text1])[0]
        embeddings2 = sentence_transformer.encode([text2])[0]
        # Calculate cosine similarity between the embeddings
        similarity_score = cosine_similarity([embeddings1], [embeddings2])[0][0]
        return similarity_score
    except Exception as e:
        logging.error(f"Error in calculate_semantic_similarity: {str(e)}")
        return 0.0

def calculate_word_overlap(text1: str, text2: str) -> float:
    """
    Calculate the word overlap ratio between two texts.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The word overlap ratio.
    """
    try:
        # Tokenize and lowercase the text
        words1 = set(preprocess_text(text1).split())
        words2 = set(preprocess_text(text2).split())
        # Calculate the word overlap ratio
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap
    except Exception as e:
        logging.error(f"Error in calculate_word_overlap: {str(e)}")
        return 0.0

def calculate_combined_similarity(text1: str, text2: str, semantic_weight: float = 0.7, overlap_weight: float = 0.3) -> float:
    """
    Calculate the combined similarity between two texts using semantic similarity and word overlap.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.
        semantic_weight (float): The weight of semantic similarity in the combined score.
        overlap_weight (float): The weight of word overlap in the combined score.

    Returns:
        float: The combined similarity score.
    """
    try:
        semantic_similarity = calculate_semantic_similarity(text1, text2)
        word_overlap = calculate_word_overlap(text1, text2)
        combined_similarity = semantic_weight * semantic_similarity + overlap_weight * word_overlap
        return combined_similarity
    except Exception as e:
        logging.error(f"Error in calculate_combined_similarity: {str(e)}")
        return 0.0