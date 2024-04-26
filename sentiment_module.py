# sentiment_module.py
from typing import List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__all__ = ['SentimentAnalyzer']

class SentimentAnalyzer:
    """
    A class for sentiment analysis using a pre-trained BERT model.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the SentimentAnalyzer with the specified model name.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
        """
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info(f"SentimentAnalyzer initialized with model: {self.model_name}")
        except Exception as e:
            logging.error(f"Error initializing SentimentAnalyzer: {str(e)}")
            self.tokenizer = None
            self.model = None

    def predict_sentiment(self, text: str) -> Optional[str]:
        """
        Predict the sentiment of the given text.

        Args:
            text (str): The input text.

        Returns:
            Optional[str]: The predicted sentiment (positive, neutral, or negative), or None if an error occurred.
        """
        if not self.tokenizer or not self.model:
            logging.error("Tokenizer or model not initialized correctly.")
            return None

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            sentiment_labels = ["negative", "neutral", "positive"]
            predicted_sentiment = sentiment_labels[predicted_class]
            logging.info(f"Predicted sentiment for '{text}': {predicted_sentiment}")
            return predicted_sentiment
        except Exception as e:
            logging.error(f"Error in predict_sentiment: {str(e)}")
            return None

    def analyze_sentiments(self, texts: List[str]) -> List[Optional[str]]:
        """
        Analyze the sentiments of multiple texts.

        Args:
            texts (List[str]): The list of input texts.

        Returns:
            List[Optional[str]]: The list of predicted sentiments (positive, neutral, or negative), or None if an error occurred.
        """
        sentiments = []
        for text in texts:
            sentiment = self.predict_sentiment(text)
            sentiments.append(sentiment)
        return sentiments