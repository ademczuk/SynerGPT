from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_sentiment(response):
    try:
        model_name = "bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()
        
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        return sentiment_labels[predicted_sentiment]
    except Exception as e:
        logging.error(f"Error in analyze_sentiment: {str(e)}")
        return "Neutral"

def check_coherence(response, chat_log):
    response_tokens = word_tokenize(response)
    chat_log_tokens = word_tokenize(chat_log)

    stop_words = set(stopwords.words('english'))
    response_tokens = [token for token in response_tokens if token.lower() not in stop_words]
    chat_log_tokens = [token for token in chat_log_tokens if token.lower() not in stop_words]

    overlap = set(response_tokens) & set(chat_log_tokens)
    overlap_ratio = len(overlap) / len(set(response_tokens))

    coherence_threshold = 0.3

    return "Coherent" if overlap_ratio >= coherence_threshold else "Not Coherent"

def evaluate_insights(response):
    vectorizer = TfidfVectorizer()
    response_vector = vectorizer.fit_transform([response])
    insight_score = response_vector.sum()

    insight_threshold = 3.0

    return "High" if insight_score >= insight_threshold else "Low"

def evaluate_relevance(response, chat_log, initial_prompt):
    vectorizer = TfidfVectorizer()
    response_vector = vectorizer.fit_transform([response])
    chat_log_vector = vectorizer.transform([chat_log])
    initial_prompt_vector = vectorizer.transform([initial_prompt])
    
    relevance_to_chat_log = cosine_similarity(response_vector, chat_log_vector)[0][0]
    relevance_to_initial_prompt = cosine_similarity(response_vector, initial_prompt_vector)[0][0]
    
    relevance_threshold = 0.4
    
    return "Relevant" if relevance_to_chat_log >= relevance_threshold and relevance_to_initial_prompt >= relevance_threshold else "Not Relevant"

def analyze_response(response, chat_log, initial_prompt):
    try:
        sentiment = analyze_sentiment(response)
        coherence = check_coherence(response, chat_log)
        insight_score = evaluate_insights(response)
        relevance = evaluate_relevance(response, chat_log, initial_prompt)
        
        feedback = f"Sentiment: {sentiment}, Coherence: {coherence}, Insight Score: {insight_score}, Relevance: {relevance}"
        return feedback
    except Exception as e:
        logging.error(f"Error in analyze_response: {str(e)}")
        return "Error analyzing response."