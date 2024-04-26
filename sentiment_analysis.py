# sentiment_analysis.py
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_sentiment(text: str) -> float:
    """
    Analyze the sentiment of the given text using NLTK's SentimentIntensityAnalyzer.

    Args:
        text (str): The input text.

    Returns:
        float: The sentiment score ranging from -1 (negative) to 1 (positive).
    """
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        return compound_score
    except Exception as e:
        logging.error(f"Error in analyze_sentiment: {str(e)}")
        return 0.0