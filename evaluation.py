# evaluation.py
from semantic_similarity import calculate_semantic_similarity
from topic_modeling import get_topic_clusters
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_conversation(conversation_history):
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