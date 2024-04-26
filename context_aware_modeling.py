# context_aware_modeling.py
import numpy as np
from utils import extract_keywords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from typing import Dict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
from gensim.models import Word2Vec
import logging
# Configure logging
#logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from logging_config import init_logger
init_logger()

# Download required NLTK resources
#nltk.download('stopwords')
#nltk.download('punkt')

# Sample data
text = ["I love this sandwich.", "This is an amazing place!", "I feel very good about these beers.", "This is my best work.", "What an awesome view", "I do not like this restaurant", "I am tired of this stuff.", "I can't deal with this", "He is my sworn enemy!", "My boss is horrible."]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Vectorizing text data
count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(text)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

class ContextAwareModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.conversation_history = []
        self.context = {}
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def update_context(self, prompt: str, response: str, feedback: str) -> None:
        """
        Update the context with the given prompt, response, and feedback.

        Args:
            prompt (str): The prompt text.
            response (str): The response text.
            feedback (str): The feedback text.
        """
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "feedback": feedback
        })
        self.vectorizer.fit([conv["response"] for conv in self.conversation_history])

    def extract_sentiment(self, feedback: str) -> str:
        """
        Extract the sentiment from the given feedback.

        Args:
            feedback (str): The feedback text.

        Returns:
            str: The extracted sentiment.
        """
        if ":" in feedback:
            parts = feedback.split(":")
            if len(parts) > 1:
                return parts[1].strip()
        return None

    def get_context(self) -> dict:
        """
        Get the current context.

        Returns:
            dict: The current context.
        """
        return self.context

    def identify_relevant_cluster(self, text, topic_clusters):
        """
        Identify the most relevant cluster based on the given text and topic clusters.

        Args:
            text (str): The text to compare against the topic clusters.
            topic_clusters (List[List[Dict]]): The topic clusters.

        Returns:
            List[Dict]: The most relevant cluster.
        """
        max_similarity = -1
        most_relevant_cluster = None
        for cluster in topic_clusters:
            cluster_texts = [conv["response"] for conv in cluster]
            cluster_text = " ".join(cluster_texts)
            try:
                similarity = cosine_similarity([text], [cluster_text])[0][0]
            except ValueError:
                # Handle the case when the input is a string
                similarity = 0.0
            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_cluster = cluster
        return most_relevant_cluster

    def get_relevant_history(self, conversation_history: List[Dict], original_prompt: str) -> List[Dict]:
        """
        Retrieve the relevant conversation history based on the original prompt.

        Args:
            conversation_history (List[Dict]): The history of the conversation.
            original_prompt (str): The original user prompt.

        Returns:
            List[Dict]: The relevant conversation history.
        """
        topic_keywords = extract_keywords(original_prompt)
        relevant_history = []

        for conv in conversation_history:
            stemmed_prompt = ' '.join([self.stemmer.stem(token) for token in conv["prompt"].lower().split() if token not in self.stop_words])
            stemmed_response = ' '.join([self.stemmer.stem(token) for token in conv["response"].lower().split() if token not in self.stop_words])
            if any(self.stemmer.stem(keyword.lower()) in stemmed_prompt or self.stemmer.stem(keyword.lower()) in stemmed_response for keyword in topic_keywords):
                relevant_history.append(conv)

        return relevant_history

    def evaluate_coherence(self, response: str, conversation_history: List[Dict]) -> float:
        """
        Evaluate the coherence of the response with respect to the conversation history using Word Mover's Distance.

        Args:
            response (str): The response text.
            conversation_history (List[Dict]): The history of the conversation.

        Returns:
            float: The coherence score.
        """
        if not conversation_history:
            return 0.0

        # Train Word2Vec model on the conversation history
        sentences = [' '.join([self.stemmer.stem(token) for token in conv['response'].lower().split() if token not in self.stop_words]) for conv in conversation_history]
        model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

        # Calculate Word Mover's Distance
        response_tokens = [self.stemmer.stem(token) for token in response.lower().split() if token not in self.stop_words]
        instance = model.wv.wmdistance(response_tokens, sentences)

        coherence_score = 1 - instance / len(response_tokens)  # Normalize the score to [0, 1] range
        return coherence_score

    def train_word2vec_model(self, conversation_history: List[Dict]) -> gensim.models.Word2Vec:
        """
        Train a Word2Vec model on the conversation history.

        Args:
            conversation_history (List[Dict]): The history of the conversation.

        Returns:
            gensim.models.Word2Vec: The trained Word2Vec model.
        """
        sentences = [' '.join([self.stemmer.stem(token) for token in conv['response'].lower().split() if token not in self.stop_words]) for conv in conversation_history]
        return Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

def test_context_aware_model():
    # Example usage
    model = ContextAwareModel()

    # Update the context with some sample data
    model.update_context("What is the capital of France?", "The capital of France is Paris.", "Sentiment: Neutral, Relevance: 5, Insightfulness: 4")
    model.update_context("What are some famous landmarks in Paris?", "Some famous landmarks in Paris include the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral.", "Sentiment: Positive, Relevance: 4, Insightfulness: 3")

    # Get the relevant conversation history
    relevant_history = model.get_relevant_history(model.conversation_history, "What are some interesting facts about Paris?")
    for conv in relevant_history:
        print(f"Prompt: {conv['prompt']}")
        print(f"Response: {conv['response']}")
        print(f"Feedback: {conv['feedback']}")
        print()

    # Evaluate the coherence of a new response
    new_response = "Paris is known for its rich history, art, and cuisine. It has been a center of culture and fashion for centuries."
    coherence_score = model.evaluate_coherence(new_response, model.conversation_history)
    print(f"Coherence score for the new response: {coherence_score:.2f}")

if __name__ == "__main__":
    test_context_aware_model()