# ANtalk.py
import os
import re
from dotenv import load_dotenv
import sys
from typing import List, Tuple
from anthropic import Client, HUMAN_PROMPT, AI_PROMPT
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from dotenv import load_dotenv
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner  # Add this line

load_dotenv(dotenv_path='_API_Keys.env')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
client = Client(api_key=ANTHROPIC_API_KEY)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClaudeManager:
    def __init__(self):
        self.model_path = './model_finetuned/'
        if os.path.exists(self.model_path):
            try:
                self.model = BertForSequenceClassification.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
            except OSError as e:
                logging.error(f"Error loading model or tokenizer: {str(e)}")
                self.model = None
                self.tokenizer = None
        else:
            logging.warning("Fine-tuned model not found. Skipping ClaudeManager initialization.")
            self.model = None
            self.tokenizer = None
    
    def generate_prompt(self, previous_response: str, conversation_history: List[dict], context_model, dynamic_planner, original_prompt: str) -> str:
        if self.model is not None:
            try:
                # Analyze sentiment of the previous response
                sentiment = self.analyze_sentiment(previous_response)
                logging.info(f"Sentiment of previous response: {sentiment}")
                
                # Retrieve relevant conversation history based on the current context
                relevant_history = context_model.get_relevant_history(conversation_history, original_prompt)
                logging.info(f"Relevant conversation history: {relevant_history}")
                
                # Generate prompt based on sentiment, relevant history, and dynamic planner
                if sentiment == "positive":
                    prompt_template = "Great! Let's continue our discussion on a positive note. Considering the relevant context:\n{relevant_history}\n{original_prompt}"
                elif sentiment == "negative":
                    prompt_template = "I understand your concerns. Let's try to approach this from a different perspective. Considering the relevant context:\n{relevant_history}\n{original_prompt}"
                else:
                    prompt_template = "Thank you for sharing your thoughts. Let's delve deeper into the topic. Considering the relevant context:\n{relevant_history}\n{original_prompt}"
                
                prompt = prompt_template.format(relevant_history=relevant_history, original_prompt=original_prompt)
                
                # Adapt the prompt based on the dynamic planner
                adapted_prompt = dynamic_planner.adapt_prompt(prompt, previous_response, conversation_history)
                logging.info(f"Adapted prompt: {adapted_prompt}")
                
                # Refine the prompt with additional information and techniques
                refined_prompt = f"USER GOAL: {context_model.user_goal}\n" \
                                 f"CONVERSATION STAGE: {context_model.current_stage}\n" \
                                 f"DESIRED OUTPUT FORMAT: {context_model.desired_format}\n" \
                                 f"{adapted_prompt}"
                
                response = client.completions.create(
                    prompt=f"{HUMAN_PROMPT} {refined_prompt}{AI_PROMPT}",
                    #stop_sequences=[HUMAN_PROMPT],
                    max_tokens_to_sample=2000,
                    model="claude-v1.3"
                )
                
                generated_response = response.completion.strip()
                logging.info(f"Generated response: {generated_response}")
                
                return generated_response
            except Exception as e:
                logging.error(f"Error generating prompt: {str(e)}")
                return "I apologize for any inconvenience. It seems I encountered an unexpected issue. Could you please provide more context or rephrase your previous response?"
        else:
            logging.warning("Fine-tuned model not loaded. Falling back to a default prompt.")
            return "I apologize for the inconvenience, but it seems my language model is not loaded correctly. Could you please provide more details about the topic you'd like to discuss? I'll do my best to assist you based on the available information."

    def analyze_sentiment(self, response: str) -> str:
        try:
            inputs = self.tokenizer(response, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()
            
            sentiment_labels = ["negative", "neutral", "positive"]
            return sentiment_labels[predicted_sentiment]
        except Exception as e:
            logging.error(f"Error in analyze_sentiment: {str(e)}")
            return "neutral"

def save_code_to_file(code: str, filename: str) -> None:
    try:
        with open(filename, 'w') as file:
            file.write(code)
        logging.info(f"Code saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving code to file {filename}: {str(e)}")

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    code_pattern = re.compile(r'```(.*?)\n(.*?)\n```', re.DOTALL)
    return code_pattern.findall(text)

def get_file_extension(language: str) -> str:
    language_extensions = {
        'python': '.py',
        'javascript': '.js', 
        'html': '.html',
        'css': '.css'
    }
    return language_extensions.get(language.lower(), '.txt')

def main():
    if len(sys.argv) < 2:
        logging.error("Please provide a prompt as a command line argument.")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    try:
        claude_manager = ClaudeManager()
        context_model = ContextAwareModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = 100
        action_dim = 4
        dynamic_planner = DynamicPlanner(state_dim, action_dim, device)

        response = client.completions.create(
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            stop_sequences=[HUMAN_PROMPT],
            max_tokens_to_sample=2000,
            model="claude-v1.3"
        )
            
        generated_response = response.completion.strip()

        code_blocks = extract_code_blocks(generated_response)
        for language, code in code_blocks:
            file_extension = get_file_extension(language)
            filename = f"code_snippet{file_extension}"
            save_code_to_file(code.strip(), filename)
        
        print(generated_response)
        logging.info(f"Generated response: {generated_response}")
        
        # Update the context model and dynamic planner based on the generated response
        context_model.update_context(prompt, generated_response, None)
        state = dynamic_planner.get_state(prompt, generated_response)
        action = dynamic_planner.select_action(state)
        reward = dynamic_planner.get_reward(prompt, generated_response)
        next_state = dynamic_planner.get_state(prompt, generated_response)
        done = 1 if reward >= 4.5 else 0
        dynamic_planner.update_model(state, action, reward, next_state, done)
        
        # Generate the next prompt based on the generated response and contextual information
        next_prompt = claude_manager.generate_prompt(generated_response, [{"prompt": prompt, "response": generated_response}], context_model, dynamic_planner, prompt)
        logging.info(f"Next prompt: {next_prompt}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()