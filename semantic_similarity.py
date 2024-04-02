# semantic_similarity.py
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

nlp = spacy.load("en_core_web_sm")
sentence_transformer = SentenceTransformer('bert-base-nli-mean-tokens')

def calculate_semantic_similarity(text1, text2):
    sentence1 = sentence_transformer.encode([text1])[0]
    sentence2 = sentence_transformer.encode([text2])[0]
    similarity_score = np.dot(sentence1, sentence2) / (np.linalg.norm(sentence1) * np.linalg.norm(sentence2))
    return similarity_score