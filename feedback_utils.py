# feedback_utils.py
import os
import torch
from transformers import BertForSequenceClassification, BertConfig, AutoTokenizer
import logging
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from sentiment_analysis import analyze_sentiment
from topic_modeling import extract_topics
from typing import Tuple
# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def alternate_feedback(llm_name: str) -> str:
    """
    Get the alternate feedback LLM name based on the current LLM name.

    Args:
        llm_name (str): The current LLM name.

    Returns:
        str: The alternate feedback LLM name.
    """
    if llm_name == "Claude":
        return "ChatGPT"
    else:
        return "Claude"

def get_llm_feedback(response: str, llm_name: str, previous_scores: Tuple[float, float]) -> Tuple[str, float, float]:
    """
    Get feedback for the LLM response using the fine-tuned model.

    Args:
        response (str): The response text.
        llm_name (str): The name of the LLM.
        previous_scores (Tuple[float, float]): The previous relevance and insightfulness scores.

    Returns:
        Tuple[str, float, float]: A tuple containing the predicted sentiment, relevance score, and insightfulness score.
    """
    try:
        model_path = './model_finetuned/'
        if os.path.exists(model_path):
            # Load the fine-tuned model and tokenizer
            config = BertConfig.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                sentiment_probs = torch.softmax(outputs.logits[:, :3], dim=1).tolist()[0]
                relevance_score = torch.sigmoid(outputs.logits[:, 3]).item()
                insightfulness_score = torch.sigmoid(outputs.logits[:, 4]).item()

            sentiment_labels = ["negative", "neutral", "positive"]
            predicted_sentiment = sentiment_labels[sentiment_probs.index(max(sentiment_probs))]

            return predicted_sentiment, relevance_score, insightfulness_score
        else:
            logging.warning("Fine-tuned model not found. Skipping sentiment analysis.")
            return "neutral", 0.0, 0.0
    except Exception as e:
        logging.error(f"Error in get_llm_feedback: {str(e)}")
        return "neutral", 0.0, 0.0

def generate_qualitative_feedback(response: str, relevance_score: float, insightfulness_score: float, factuality_score: float, reasoning_score: float) -> str:
    """
    Generate qualitative feedback based on the response, relevance score, insightfulness score, factuality score, and reasoning score.
    Args:
        response (str): The response text.
        relevance_score (float): The relevance score.
        insightfulness_score (float): The insightfulness score.
        factuality_score (float): The factuality score.
        reasoning_score (float): The reasoning score.
    Returns:
        str: The generated qualitative feedback.
    """
    # Tokenize the response
    sentences = sent_tokenize(response)
    tokens = word_tokenize(response.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]

    # Calculate response length and unique word count
    response_length = len(filtered_tokens)
    unique_words = set(filtered_tokens)
    unique_word_count = len(unique_words)

    # Evaluate relevance
    relevance_feedback = []
    if relevance_score < 3:
        relevance_feedback.append("The response could have been more relevant to the topic.")
    else:
        relevance_feedback.append("The response was relevant to the topic.")

    # Evaluate insightfulness
    insightfulness_feedback = []
    if insightfulness_score < 3:
        insightfulness_feedback.append("The response lacked unique insights and perspectives.")
    else:
        insightfulness_feedback.append("The response provided unique insights and perspectives.")

    # Evaluate response length
    length_feedback = []
    if response_length < 50:
        length_feedback.append("The response was quite brief. Consider providing more detailed explanations.")
    elif response_length > 500:
        length_feedback.append("The response was quite lengthy. Consider being more concise.")
    else:
        length_feedback.append("The response was of an appropriate length.")

    # Evaluate vocabulary diversity
    vocabulary_feedback = []
    unique_word_ratio = unique_word_count / response_length
    if unique_word_ratio < 0.3:
        vocabulary_feedback.append("The response could benefit from using a more diverse vocabulary.")
    else:
        vocabulary_feedback.append("The response exhibited a good vocabulary diversity.")

    # Evaluate factuality
    factuality_feedback = []
    if factuality_score < 0.5:
        factuality_feedback.append("The response contains some inaccurate or unverified information. Please double-check the facts.")
    else:
        factuality_feedback.append("The response appears to be factually accurate based on the available information.")
    
    # Evaluate reasoning
    reasoning_feedback = []
    if reasoning_score < 0.5:
        reasoning_feedback.append("The response lacks logical reasoning and coherent arguments. Try to provide more well-reasoned explanations.")
    else:
        reasoning_feedback.append("The response demonstrates sound reasoning and presents coherent arguments.")
    
    # Analyze sentiment
    sentiment_feedback = []
    sentiment_score = analyze_sentiment(response)
    if sentiment_score < -0.3:
        sentiment_feedback.append("The response had a negative sentiment. Consider using a more neutral or positive tone.")
    elif sentiment_score > 0.3:
        sentiment_feedback.append("The response had a positive sentiment. Maintain the encouraging and constructive tone.")
    else:
        sentiment_feedback.append("The response had a neutral sentiment.")

    # Extract topics
    topics = extract_topics(response)
    topic_feedback = [f"The response covered the following topics: {', '.join(topics)}."]

    # Combine feedback
    feedback = "\n".join(relevance_feedback + insightfulness_feedback + length_feedback + vocabulary_feedback + sentiment_feedback + topic_feedback + factuality_feedback + reasoning_feedback)
    return feedback

def save_to_labeled_dataset(response: str, feedback: str, context_model: ContextAwareModel, responder_name: str, dynamic_planner: DynamicPlanner) -> None:
    """
    Save the response, feedback, context, responder name, and dynamic plan to the labeled dataset.

    Args:
        response (str): The response text.
        feedback (str): The feedback text.
        context_model (ContextAwareModel): The context-aware modeling object.
        responder_name (str): The name of the responder.
        dynamic_planner (DynamicPlanner): The dynamic planner object.
    """
    try:
        with open('labeled_dataset.json', 'r') as file:
            labeled_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        labeled_data = []
    
    context = context_model.get_context()
    last_context = context.get('topic', [])
    dynamic_plan = dynamic_planner.get_current_plan() if dynamic_planner else []
    
    labeled_data.append({
        "text": response.replace('\\\\', '\\'),
        "feedback": feedback.replace('\\\\', '\\'),
        "context": last_context,
        "responder_name": responder_name,
        "dynamic_plan": dynamic_plan
    })
    
    try:
        with open('labeled_dataset.json', 'w') as file:
            json.dump(labeled_data, file, indent=2)
        logging.info("Labeled dataset saved successfully.")
    except Exception as e:
        logging.error(f"Error saving to labeled dataset: {str(e)}")

def initialize_labeled_dataset() -> None:
    """
    Initialize the labeled dataset file if it doesn't exist or is invalid.
    """
    try:
        with open('labeled_dataset.json', 'r') as file:
            json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        with open('labeled_dataset.json', 'w') as file:
            json.dump([], file)