# fallback_models.py
from typing import List
from utils import extract_keywords
from transformers import pipeline

class FallbackModel:
    def __init__(self):
        # Initialize a pre-trained language model for more engaging responses
        self.generator = pipeline('text-generation', model='gpt2')  # You can choose other models

    def generate_prompt(self, previous_response: str, conversation_history: List[dict], context_model, original_prompt: str) -> str:
        # Combine keyword-based and pre-trained model approaches
        keywords = extract_keywords(previous_response)

        if keywords:
            # Generate prompt using keywords and context
            topic = ', '.join(keywords)
            context = context_model.get_relevant_context(conversation_history, keywords)
            if context:
                prompt_text = f"Considering the topics: {topic} and the previous discussion about {context}, what else would you like to explore related to the initial question: {original_prompt}?"
            else:
                prompt_text = f"Let's talk more about {topic} in relation to the original prompt: {original_prompt}."
        else:
            # Use the pre-trained model to generate a more open-ended prompt
            prompt_text = f"Based on our previous conversation, what interesting aspects related to '{original_prompt}' would you like to discuss further?"

        # Use the pre-trained model for better fluency and creativity
        fallback_prompt = self.generator(prompt_text, max_length=50, num_return_sequences=1)[0]['generated_text']
        return fallback_prompt