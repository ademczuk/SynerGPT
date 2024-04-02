# context_aware_modeling.py
from topic_modeling import get_topic_clusters
from semantic_similarity import calculate_semantic_similarity

class ContextAwareModel:
    def __init__(self):
        self.conversation_history = []
        self.context = {}

    def update_context(self, prompt, response, feedback):
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "feedback": feedback
        })

        topic_clusters = get_topic_clusters(self.conversation_history)
        most_relevant_cluster = self.identify_relevant_cluster(response, topic_clusters)

        self.context = {
            "topic": [conv["response"] for conv in most_relevant_cluster],
            "sentiment": self.extract_sentiment(feedback) if feedback else None
        }

    def extract_sentiment(self, feedback):
        if ":" in feedback:
            parts = feedback.split(":")
            if len(parts) > 1:
                return parts[1].strip()
        return None

    def get_context(self):
        return self.context

    def identify_relevant_cluster(self, text, topic_clusters):
        max_similarity = -1
        most_relevant_cluster = None

        for cluster in topic_clusters:
            cluster_texts = [conv["response"] for conv in cluster]
            cluster_text = " ".join(cluster_texts)
            similarity = calculate_semantic_similarity(text, cluster_text)
            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_cluster = cluster

        return most_relevant_cluster

    def get_relevant_history(self, conversation_history, topic_keywords):
        relevant_history = []
        for conv in conversation_history:
            if any(keyword in conv["prompt"].lower() or keyword in conv["response"].lower() for keyword in topic_keywords):
                relevant_history.append(conv)
        return relevant_history