# analyze_response.py
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from question_answering import answer_question
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from reinforcement_learning import evaluate_insights

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")
def analyze_sentiment(response: str) -> str:
    try:
        model_name = "finetuned_sentiment_model"  # Replace with your fine-tuned model path
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()
        
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        sentiment = sentiment_labels[predicted_sentiment]
        logging.info("Sentiment analysis result: {sentiment}")
        return sentiment
    except Exception as e:
        logging.error("Error in analyze_sentiment: {str(e)}")
        return "Neutral"
def check_coherence(response: str, chat_log: str) -> str:
    response_tokens = word_tokenize(response)
    chat_log_tokens = word_tokenize(chat_log)
    stop_words = set(stopwords.words('english'))
    response_tokens = [token for token in response_tokens if token.lower() not in stop_words]
    chat_log_tokens = [token for token in chat_log_tokens if token.lower() not in stop_words]
    
    # Train Word2Vec model on chat_log_tokens
    model = Word2Vec([chat_log_tokens], vector_size=100, window=5, min_count=1, workers=4)
    
    # Calculate Word Mover's Distance
    instance = WmdSimilarity([response_tokens], model.wv)
    coherence_score = instance[chat_log_tokens][0]
    
    coherence_threshold = 0.5
    coherence = "Coherent" if coherence_score >= coherence_threshold else "Not Coherent"
    logging.info("Coherence check result: {coherence} (WMD Score: {coherence_score:.2f})")
    return coherence

def evaluate_relevance(response: str, chat_log: str, initial_prompt: str) -> str:
    try:
        # Create BM25 object
        corpus = [chat_log, initial_prompt]
        bm25_obj = BM25Okapi(corpus)
        
        # Calculate relevance scores using BM25
        relevance_to_chat_log = bm25_obj.get_scores([response])[0]
        relevance_to_initial_prompt = bm25_obj.get_scores([response])[1]
        
        relevance_threshold = 2.0
        relevance = "Relevant" if (relevance_to_chat_log >= relevance_threshold and
                                   relevance_to_initial_prompt >= relevance_threshold) else "Not Relevant"
        logging.info("Relevance evaluation result: {relevance} (Chat log relevance: {relevance_to_chat_log:.2f}, "
                     "Initial prompt relevance: {relevance_to_initial_prompt:.2f})")
        return relevance
    except Exception as e:
        logging.error("Error in evaluate_relevance: {str(e)}")
        return "Not Relevant"
    
def analyze_response(response: str, chat_log: str, initial_prompt: str) -> str:
    try:
        sentiment = analyze_sentiment(response)
        coherence = check_coherence(response, chat_log)
        insight_score = evaluate_insights(response)
        relevance = evaluate_relevance(response, chat_log, initial_prompt)
        
        # Perform question answering
        question = "What is the main topic of the response?"
        answer = answer_question(question, response)
        
        feedback = "Sentiment: {sentiment}, Coherence: {coherence}, Insight Score: {insight_score}, Relevance: {relevance}, Main Topic: {answer}"
        logging.info("Response analysis feedback: {feedback}")
        return feedback
    except Exception as e:
        logging.error("Error in analyze_response: {str(e)}")
        return "Error analyzing response."
    
def evaluate_sentiment_analysis(true_sentiments: List[int], predicted_sentiments: List[int]) -> Tuple[float, float, float, float]:
    try:
        accuracy = accuracy_score(true_sentiments, predicted_sentiments)
        precision = precision_score(true_sentiments, predicted_sentiments, average='weighted')
        recall = recall_score(true_sentiments, predicted_sentiments, average='weighted')
        f1 = f1_score(true_sentiments, predicted_sentiments, average='weighted')
        
        logging.info("Sentiment Analysis Evaluation:")
        logging.info("Accuracy: {accuracy:.4f}")
        logging.info("Precision: {precision:.4f}")
        logging.info("Recall: {recall:.4f}")
        logging.info("F1 Score: {f1:.4f}")
        
        return accuracy, precision, recall, f1
    except Exception as e:
        logging.error("Error in evaluate_sentiment_analysis: {str(e)}")
        return None, None, None, None