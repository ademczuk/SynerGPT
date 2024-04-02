import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
import json
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def alternate_feedback(llm_name):
    if llm_name == "Claude":
        return "ChatGPT"
    else:
        return "Claude"

def get_llm_feedback(response, llm_name, previous_scores):
    try:
        model_path = './model_finetuned/'
        if os.path.exists(model_path):
            model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
            tokenizer = BertTokenizer.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                sentiment_probs = torch.softmax(outputs.logits[:, :3], dim=1).tolist()[0]
                relevance_score = torch.sigmoid(outputs.logits[:, 0]).item()
                insightfulness_score = torch.sigmoid(outputs.logits[:, 1]).item()

            sentiment_labels = ["negative", "neutral", "positive"]
            predicted_sentiment = sentiment_labels[sentiment_probs.index(max(sentiment_probs))]

            # Dynamic scoring and feedback
            previous_relevance, previous_insightfulness = previous_scores
            relevance_improvement = relevance_score - previous_relevance
            insightfulness_improvement = insightfulness_score - previous_insightfulness

            if relevance_improvement > 0 or insightfulness_improvement > 0:
                progress_points = int(relevance_improvement * 10) + int(insightfulness_improvement * 10)
                logging.info(f"{llm_name} earned {progress_points} progress points for improvement!")

            if relevance_score >= 4.5 and insightfulness_score >= 4.5:
                logging.info(f"Congratulations, {llm_name}! You have achieved a high level of relevance and insightfulness.")
                # Implement logic to unlock new levels or abilities

            qualitative_feedback = generate_qualitative_feedback(response, relevance_score, insightfulness_score)
            logging.info(f"Qualitative feedback for {llm_name}: {qualitative_feedback}")

            logging.info(f"{llm_name}'s feedback on the response:")
            logging.info(f"Sentiment: {predicted_sentiment}")
            logging.info(f"Relevance: {relevance_score:.2f}")
            logging.info(f"Insightfulness: {insightfulness_score:.2f}")

            return predicted_sentiment, relevance_score, insightfulness_score
        else:
            logging.warning("Fine-tuned model not found. Skipping sentiment analysis.")
            return "neutral", 0.0, 0.0

    except Exception as e:
        logging.error(f"Error in get_llm_feedback: {str(e)}")
        return "neutral", 0.0, 0.0

def generate_qualitative_feedback(response, relevance_score, insightfulness_score):
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

    # Combine feedback
    feedback = "\n".join(relevance_feedback + insightfulness_feedback + length_feedback + vocabulary_feedback)

    return feedback

def save_to_labeled_dataset(response, feedback, context_model, responder_name, dynamic_planner):
    try:
        with open('labeled_dataset.json', 'r') as file:
            labeled_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        labeled_data = []

    context = context_model.get_context()
    last_context = context.get('topic', [])  # Get the 'topic' key from the context dictionary, default to an empty list if not present

    labeled_data.append({
        "text": response.replace('\\\\', '\\'),
        "feedback": feedback.replace('\\\\', '\\'),
        "context": last_context,
        "responder_name": responder_name,
        "dynamic_plan": dynamic_planner.get_current_plan()
    })

    try:
        with open('labeled_dataset.json', 'w') as file:
            json.dump(labeled_data, file, indent=2)
    except Exception as e:
        logging.error(f"Error saving to labeled dataset: {str(e)}")