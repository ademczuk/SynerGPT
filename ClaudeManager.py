# ClaudeManager.py
from reinforcement_learning import DynamicPlanner
import os
import torch
import logging
import random
from transformers import BertTokenizer, BertForSequenceClassification
from context_aware_modeling import ContextAwareModel
from typing import List
from fallback_models import FallbackModel

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Use __name__


class ClaudeManager:
    def __init__(self, dynamic_planner, model_path='./model_finetuned/'):
        self.model_path = model_path
        self.is_model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamic_planner = dynamic_planner  # Store the dynamic_planner instance

        self.load_model()
        self.fallback_model = FallbackModel()  # Initialize the fallback model 

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = BertForSequenceClassification.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.is_model_loaded = True
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self.is_model_loaded = False
        else:
            logger.warning("Model path does not exist")
            self.is_model_loaded = False 

    def assign_role(self) -> str:
        roles = ["Devil's Advocate", "Optimist", "Skeptic", "Futurist", "Realist", "Innovator", "Critic", "Visionary"]
        return random.choice(roles)

    def generate_prompt(self, previous_response: str, conversation_history: List[dict], context_model: ContextAwareModel, original_prompt: str) -> str:
        if self.is_model_loaded:
            # Fine-tuned model available
            adapted_prompt = self.dynamic_planner.adapt_prompt(previous_response, conversation_history)
            logger.info(f"Adapted prompt: {adapted_prompt}")

            # Assuming you have a 'generate_prompt' function for your fine-tuned model
            # (This is just an example, you might have your custom logic)
            new_prompt = f"Based on the adapted prompt: '{adapted_prompt}', generate a creative and informative response." 
            
            return new_prompt
        else:
            # Fallback to FallbackModel
            logger.warning("Fine-tuned model not loaded. Using fallback model.")
            fallback_prompt = self.fallback_model.generate_prompt(previous_response, conversation_history, context_model, original_prompt)
        return fallback_prompt