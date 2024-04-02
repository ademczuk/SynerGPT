import os
import re
import sys
from typing import List, Tuple
from anthropic import Client, HUMAN_PROMPT, AI_PROMPT
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from dotenv import load_dotenv

load_dotenv()
anthropic_api_key = os.environ.get('anthropic_api_key')
client = Client(api_key=anthropic_api_key)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClaudeManager:
    def __init__(self):
        self.model_path = './model_finetuned/'
        if os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            logging.warning("Fine-tuned model not found. Skipping ClaudeManager initialization.")
            self.model = None
            self.tokenizer = None

    def generate_prompt(self, previous_response):
        # Use the fine-tuned BERT model to analyze the previous response
        # and generate a new prompt based on the analysis
        if self.model is not None:
            # Implement your prompt generation logic here
            new_prompt = "Please elaborate on your previous point."
        else:
            new_prompt = "Please provide more details on the topic."
        return new_prompt

def save_code_to_file(code: str, filename: str) -> None:
    try:
        with open(filename, 'w') as file:
            file.write(code)
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
        new_prompt = prompt  # Initialize new_prompt with the provided prompt
        if os.path.exists('./model_finetuned/'):
            claude_manager = ClaudeManager()
            new_prompt = claude_manager.generate_prompt(prompt)

        response = client.completions.create(
            prompt=f"{HUMAN_PROMPT} {new_prompt}{AI_PROMPT}",
            max_tokens_to_sample=550,
            model="claude-v1"
        )

        generated_response = response.completion.strip()

        code_blocks = extract_code_blocks(generated_response)
        for language, code in code_blocks:
            file_extension = get_file_extension(language)
            filename = f"code_snippet{file_extension}"
            save_code_to_file(code.strip(), filename)
            logging.info(f"Code saved to {filename}")

        print(generated_response)  # Print the generated response to the terminal
        logging.info(generated_response)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()