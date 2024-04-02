import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
import random

from topic_modeling import get_topic_clusters
from semantic_similarity import calculate_semantic_similarity
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClaudeManager:
    def __init__(self):
        self.model_path = './model_finetuned/'
        self.is_model_loaded = False
        if os.path.exists(self.model_path):
            try:
                self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=3)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.is_model_loaded = True
            except OSError as e:
                logging.error(f"Error loading model or tokenizer: {str(e)}")
                self.model = None
                self.tokenizer = None
        else:
            logging.warning("Fine-tuned model not found. Skipping ClaudeManager initialization.")
            self.model = None
            self.tokenizer = None

    def generate_prompt(self, previous_response, conversation_history, context_model, dynamic_planner):
        # Use the fine-tuned BERT model to analyze the previous response
        # and generate a new prompt based on the analysis
        if self.is_model_loaded:
            # Implement your prompt generation logic here
            new_prompt = "Please elaborate on your previous point."
            # Incorporate relevant information from conversation history
            relevant_history = self.get_relevant_history(conversation_history[-1:], previous_response, context_model)  # Pass only the last conversation entry
            if relevant_history:
                new_prompt += f"\nAdditional context from previous conversation:\n{relevant_history}"
            # Add open-ended questions and encourage exploration of multiple perspectives
            new_prompt += "\nConsider the following questions:\n"
            new_prompt += "1. What are the potential implications of this idea?\n"
            new_prompt += "2. How can we approach this from a different angle?\n"
            new_prompt += "3. What evidence supports or challenges this perspective?"
            # Use dynamic planning to adapt the conversation flow
            new_prompt = dynamic_planner.adapt_prompt(new_prompt, previous_response, conversation_history)
        else:
            new_prompt = "Please provide more details on the topic."
        return new_prompt

    def get_relevant_history(self, conversation_history, previous_response, context_model):
        # Implement logic to retrieve relevant information from conversation history based on previous response
        # This can be done using techniques like topic modeling or semantic similarity
        # Return the relevant portion of the conversation history
        topic_clusters = get_topic_clusters(conversation_history)
        if not topic_clusters:
            return ""  # Return an empty string if topic_clusters is empty

        relevant_cluster = context_model.identify_relevant_cluster(previous_response, topic_clusters)
        if not relevant_cluster:
            return ""  # Return an empty string if relevant_cluster is empty

        relevant_history = "\n".join([f"Prompt: {conv['prompt']}\nResponse: {conv['response'].replace('\\\\', '\\')}"
                                      for conv in relevant_cluster])
        return relevant_history

    def assign_role(self):
        roles = ["Devil's Advocate", "Optimist", "Skeptic", "Futurist"]
        return random.choice(roles)