# ChatGPTWorker.py
import openai
import os
from dotenv import load_dotenv
import re
import sys
from typing import List, Tuple
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import logging
from config import LOG_LEVEL, LOG_FORMAT
import json
from dotenv import load_dotenv
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from ClaudeManager import ClaudeManager
from ANtalk import extract_code_blocks, get_file_extension, save_code_to_file
from anthropic import Client, HUMAN_PROMPT, AI_PROMPT
from feedback_utils import get_llm_feedback, save_to_labeled_dataset, initialize_labeled_dataset
from refine_prompt import refine_prompt
from utils import UpdateModelData

# Load environment variables from the _API_Keys.env file


load_dotenv(dotenv_path='_API_Keys.env')
openai.api_key = os.getenv('OPENAI_API_KEY')
client = Client(api_key=openai.api_key)

# Configure logging
logging.basicConfig(filename='app.log', level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ChatGPTWorker:
    def __init__(self, model_path='C:\\chromedriver\\Script\\Chat2Chat\\SynerGPT\\model_finetuned\\'):
        self.model_path = model_path
        print(f"Model path: {self.model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create the model_finetuned directory and its required files
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
            # Create config.json
            config = BertConfig()
            config.save_pretrained(self.model_path)
            
            # Create dummy tokenizer.json
            dummy_tokenizer = {
                "model": {
                    "vocab_size": 30522,
                    "pad_token_id": 0,
                    "type_vocab_size": 2
                }
            }
            with open(os.path.join(self.model_path, 'tokenizer.json'), 'w') as tokenizer_file:
                json.dump(dummy_tokenizer, tokenizer_file)
            
            # Create dummy pytorch_model.bin
            with open(os.path.join(self.model_path, 'pytorch_model.bin'), 'wb') as model_file:
                model_file.write(b'dummy_model_data')
        
        self.load_model()
        self.context_model = ContextAwareModel()
        self.dynamic_planner = DynamicPlanner(state_dim=100, action_dim=4, device=self.device)

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                if not all(os.path.exists(os.path.join(self.model_path, file)) for file in required_files):
                    raise FileNotFoundError(f"Required files not found in {self.model_path}")
                
                self.model = BertForSequenceClassification.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model.to(self.device)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model directory not found: {self.model_path}")
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_response(self, prompt: str, role: str, context_model: ContextAwareModel, dynamic_planner: DynamicPlanner) -> str:
        """
        Generate a response using OpenAI's GPT-3 model.

        Args:
            prompt (str): The prompt text.
            role (str): The role of the responder.
            context_model (ContextAwareModel): The context-aware modeling object.
            dynamic_planner (DynamicPlanner): The dynamic planner object.

        Returns:
            str: The generated response.
        """
        try:
            # Send prompt to GPT-3 and receive response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": role, "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()

            # Initialize sentiment with a default value
            sentiment = "neutral"

            if self.model is not None:
                # Perform sentiment analysis using the fine-tuned model
                inputs = self.tokenizer(response_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()
                sentiment_labels = ["negative", "neutral", "positive"]
                sentiment = sentiment_labels[predicted_sentiment]
                logging.info(f"Sentiment analysis: {sentiment}")  # Log the predicted sentiment

            # Update the context model with the sentiment
            context_model.update_context(prompt, response_text, sentiment)

            # Update the dynamic planner
            state = dynamic_planner.get_state(prompt, response_text)
            action = dynamic_planner.select_action(state)
            reward = dynamic_planner.get_reward(prompt, response_text)
            next_state = dynamic_planner.get_state(prompt, response_text)
            done = 1 if reward >= 4.5 else 0
            dynamic_planner.update_model(state, action, reward, next_state, done)

            return response_text
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return ""
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return ""

def guess_language(code_snippet: str) -> str:
    """
    Guess the programming language based on the code snippet.
    Args:
        code_snippet (str): The code snippet to analyze.
    Returns:
        str: The guessed programming language.
    """
    language_patterns = [
        (r'<html', 'html'),
        (r'{|}', 'js'),
        (r'@import|{', 'css'),
        (r'def |import |class ', 'py')
    ]
    for pattern, language in language_patterns:
        if re.search(pattern, code_snippet, re.IGNORECASE):
            return language
    return 'txt'

def extract_code(text: str) -> List[Tuple[str, str]]:
    """
    Extract code snippets from the given text.
    Args:
        text (str): The text to extract code snippets from.
    Returns:
        List[Tuple[str, str]]: A list of tuples containing the language and code snippet for each code block.
    """
    code_pattern = re.compile(r'```(.*?)\n(.*?)\n```', re.DOTALL)
    return code_pattern.findall(text)

def simplify_prompt(prompt: str) -> str:
    """
    Simplify the prompt by removing non-alphanumeric characters and limiting its length.
    Args:
        prompt (str): The prompt text to simplify.
    Returns:
        str: The simplified prompt text.
    """
    return re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:30]

def main(prompt: str):
    try:
        chatgpt_worker = ChatGPTWorker()
        claude_manager = ClaudeManager()
        context_model = ContextAwareModel()
        dynamic_planner = DynamicPlanner(state_dim=100, action_dim=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        max_cycles = 10  # Define max_cycles
        cycle_count = 0
        previous_scores = {"Claude": (0.0, 0.0), "ChatGPT": (0.0, 0.0)}
        response = ""
        conversation_history = []  # Define conversation_history
        original_prompt = prompt  # Define original_prompt

        while cycle_count < max_cycles:
            if cycle_count % 2 == 0:
                role = claude_manager.assign_role()
                # Update the function calls to include dynamic_planner
                refined_prompt = refine_prompt(response, "ChatGPT", conversation_history[-1:], context_model, original_prompt, claude_manager.model, claude_manager.tokenizer, claude_manager.device, dynamic_planner)
                
                if claude_manager.is_model_loaded:
                    response = claude_manager.generate_prompt(previous_response=response, conversation_history=conversation_history, context_model=context_model, dynamic_planner=dynamic_planner, original_prompt=original_prompt)
                else:
                    response = "Please provide more details on the topic."
                
                chatgpt_worker_response = conversation_history[-1]['response'] if conversation_history else ""
                responder_name = "ClaudeManager"
            else:
                role = "Assistant"
                refined_prompt = refine_prompt(response, "ChatGPT", conversation_history[-1:], context_model, original_prompt, claude_manager.model, claude_manager.tokenizer, claude_manager.device)  # Pass the required arguments for refine_prompt
                response = chatgpt_worker.generate_response(refined_prompt, role, context_model, dynamic_planner)
                claude_manager_response = conversation_history[-1]['response'] if conversation_history else ""
                responder_name = "ChatGPTWorker"
            
            llm_name = "Claude" if cycle_count % 2 == 0 else "ChatGPT"
            previous_relevance, previous_insightfulness = previous_scores[llm_name]
            predicted_sentiment, relevance_score, insightfulness_score = get_llm_feedback(response, llm_name, (previous_relevance, previous_insightfulness))
            previous_scores[llm_name] = (relevance_score, insightfulness_score)

            # Save feedback to labeled dataset
            if cycle_count % 2 == 0:
                save_to_labeled_dataset(chatgpt_worker_response, "", context_model, responder_name, dynamic_planner)  # Pass dynamic_planner
            else:
                save_to_labeled_dataset(claude_manager_response, "", context_model, responder_name, dynamic_planner)  # Pass dynamic_planner

            conversation_history.append({"prompt": refined_prompt, "response": response})
            cycle_count += 1

        response_text = chatgpt_worker.generate_response(prompt, role, context_model, dynamic_planner)
        
        print(response_text)  # Print the generated response to the terminal
        logging.info(response_text)
        
        code_snippets = extract_code(response_text)
        if code_snippets:
            simplified_prompt = simplify_prompt(prompt)
            for i, (lang, snippet) in enumerate(code_snippets, start=1):
                filename = f"{simplified_prompt}_{i}.{guess_language(snippet)}"
                try:
                    with open(filename, "w") as code_file:
                        code_file.write(snippet.strip())
                        logging.info(f"Code snippet saved to {filename}")
                except Exception as e:
                    logging.error(f"Error saving code snippet to {filename}: {str(e)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("No prompt provided. Usage: python script.py 'Your prompt here'")
        sys.exit(1)

    user_prompt = ' '.join(sys.argv[1:])
    main(user_prompt)