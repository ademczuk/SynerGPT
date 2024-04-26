# evaluation.py
import logging
from semantic_similarity import calculate_semantic_similarity
from topic_modeling import get_topic_clusters
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
import requests  # or whichever library you need for API integration

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_factuality(response: str) -> float:
    """
    Evaluate the factuality score of the response.
    Args:
        response (str): The response text.
    Returns:
        float: The factuality score.
    """
    try:
        # Integrate with a fact-checking API here
        # For example, send the response to the API and retrieve the factuality score
        api_response = requests.post('https://factcheckingapi.com/check', data={'text': response})
        factuality_score = api_response.json()['factuality_score']
        return factuality_score
    except Exception as e:
        logging.error(f"Error in evaluate_factuality: {str(e)}")
        return 0.0

def evaluate_reasoning(response: str) -> float:
    """
    Evaluate the reasoning score of the response.
    Args:
        response (str): The response text.
    Returns:
        float: The reasoning score.
    """
    try:
        # Integrate with a reasoning evaluation API or service here
        # For example, send the response to the API and retrieve the reasoning score
        api_response = requests.post('https://reasoningapi.com/evaluate', data={'text': response})
        reasoning_score = api_response.json()['reasoning_score']
        return reasoning_score
    except Exception as e:
        logging.error(f"Error in evaluate_reasoning: {str(e)}")
        return 0.0

def evaluate_conversation(conversation_history: List[Dict[str, str]]) -> Tuple[float, float]:
    """
    Evaluate the conversation based on relevance and coherence scores.

    Args:
        conversation_history (List[Dict[str, str]]): The history of the conversation.

    Returns:
        Tuple[float, float]: A tuple containing the average relevance score and average coherence score.
    """
    topic_clusters = get_topic_clusters(conversation_history)
    relevance_scores = []
    coherence_scores = []

    for i in range(len(conversation_history)):
        current_conv = conversation_history[i]
        prev_conv = conversation_history[i - 1] if i > 0 else None
        next_conv = conversation_history[i + 1] if i < len(conversation_history) - 1 else None

        # Calculate relevance score
        relevant_cluster = None
        max_similarity = -1
        for cluster in topic_clusters:
            cluster_texts = [conv["response"] for conv in cluster]
            cluster_text = " ".join(cluster_texts)
            similarity = calculate_semantic_similarity(current_conv["response"], cluster_text)
            if similarity > max_similarity:
                max_similarity = similarity
                relevant_cluster = cluster
        relevance_score = max_similarity

        # Calculate coherence score
        coherence_score = 0
        if prev_conv:
            coherence_score += calculate_semantic_similarity(current_conv["response"], prev_conv["response"])
        if next_conv:
            coherence_score += calculate_semantic_similarity(current_conv["response"], next_conv["response"])
        coherence_score /= 2  # Average of similarities with previous and next conversations

        relevance_scores.append(relevance_score)
        coherence_scores.append(coherence_score)

    avg_relevance_score = sum(relevance_scores) / len(relevance_scores)
    avg_coherence_score = sum(coherence_scores) / len(coherence_scores)

    logging.info(f"Conversation Evaluation:")
    logging.info(f"Average Relevance Score: {avg_relevance_score:.2f}")
    logging.info(f"Average Coherence Score: {avg_coherence_score:.2f}")

    return avg_relevance_score, avg_coherence_score

def evaluate_sentiment_classification(true_sentiments: List[int], predicted_sentiments: List[int]) -> Tuple[float, float, float, float]:
    """
    Evaluate the sentiment classification performance.

    Args:
        true_sentiments (List[int]): List of true sentiment labels.
        predicted_sentiments (List[int]): List of predicted sentiment labels.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(true_sentiments, predicted_sentiments)
    precision = precision_score(true_sentiments, predicted_sentiments, average='weighted')
    recall = recall_score(true_sentiments, predicted_sentiments, average='weighted')
    f1 = f1_score(true_sentiments, predicted_sentiments, average='weighted')

    logging.info(f"Sentiment Classification Evaluation:")
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1-score: {f1:.2f}")

    return accuracy, precision, recall, f1

def evaluate_topic_modeling(conversation_history: List[Dict[str, str]], num_topics: int = 5) -> Tuple[int, float]:
    """
    Evaluate the topic modeling performance.

    Args:
        conversation_history (List[Dict[str, str]]): The history of the conversation.
        num_topics (int): The expected number of topics.

    Returns:
        Tuple[int, float]: A tuple containing the number of topic clusters and average topic coherence score.
    """
    topic_clusters = get_topic_clusters(conversation_history)
    if len(topic_clusters) != num_topics:
        logging.warning(f"Number of topic clusters ({len(topic_clusters)}) does not match the expected number of topics ({num_topics}).")

    topic_coherence_scores = []
    for cluster in topic_clusters:
        cluster_texts = [conv["response"] for conv in cluster]
        cluster_text = " ".join(cluster_texts)
        coherence_score = calculate_semantic_similarity(cluster_text, cluster_text)
        topic_coherence_scores.append(coherence_score)

    avg_topic_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)

    logging.info(f"Topic Modeling Evaluation:")
    logging.info(f"Number of Topic Clusters: {len(topic_clusters)}")
    logging.info(f"Average Topic Coherence Score: {avg_topic_coherence:.2f}")

    return len(topic_clusters), avg_topic_coherence