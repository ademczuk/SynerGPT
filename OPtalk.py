# Optalk.py
import openai
from dotenv import load_dotenv
import os
import re
import sys
from typing import List, Tuple
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path='_API_Keys.env')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatGPTWorker:
    def __init__(self):
        self.model_path = './model_finetuned/'
        if os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            logging.warning("Fine-tuned model not found. Skipping ChatGPTWorker initialization.")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using OpenAI's GPT-3 model.

        Args:
            prompt (str): The prompt text.

        Returns:
            str: The generated response.
        """
        try:
            # Send prompt to GPT-3 and receive response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            
            if self.model is not None:
                # Perform sentiment analysis using the fine-tuned model
                inputs = self.tokenizer(response_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()
                sentiment_labels = ["negative", "neutral", "positive"]
                logging.info(f"Sentiment analysis: {sentiment_labels[predicted_sentiment]}")  # Log the predicted sentiment
            
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
        code_snippet (str): The code snippet.

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
        text (str): The text containing code snippets.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the language and code snippet.
    """
    code_pattern = re.compile(r'```(.*?)\n(.*?)\n```', re.DOTALL)
    return code_pattern.findall(text)

def simplify_prompt(prompt: str) -> str:
    """
    Simplify the prompt by removing non-alphanumeric characters and limiting its length.

    Args:
        prompt (str): The prompt text.

    Returns:
        str: The simplified prompt.
    """
    return re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:30]

def main():
    if len(sys.argv) < 2:
        logging.error("No prompt provided. Usage: python script.py 'Your prompt here'")
        sys.exit(1)
    
    user_prompt = ' '.join(sys.argv[1:])
    
    try:
        chatgpt_worker = ChatGPTWorker()
        response_text = chatgpt_worker.generate_response(user_prompt)
        
        print(response_text)  # Print the generated response to the terminal
        logging.info(response_text)
        
        code_snippets = extract_code(response_text)
        if code_snippets:
            simplified_prompt = simplify_prompt(user_prompt)
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
    main()