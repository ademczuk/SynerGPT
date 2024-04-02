import openai
import os
import re
import sys
from typing import List, Tuple
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('openai.api_key')

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatGPTWorker:
    def __init__(self):
        self.model_path = './model_finetuned/'
        if os.path.exists(self.model_path):
            try:
                self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=3)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
            except OSError as e:
                logging.error(f"Error loading model or tokenizer: {str(e)}")
                self.model = None
                self.tokenizer = None
        else:
            logging.warning("Fine-tuned model not found. Skipping ChatGPTWorker initialization.")
            self.model = None
            self.tokenizer = None

    def generate_response(self, prompt, role, context_model, dynamic_planner):
        try:
            # Send prompt to GPT-3 and receive response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. Your current role is: {role}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            response_text = response.choices[0].message['content']

            if self.model is not None:
                # Perform sentiment analysis using the fine-tuned model
                inputs = self.tokenizer(response_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                predicted_sentiment = torch.argmax(outputs.logits[:, :3], dim=1).item()
                sentiment_labels = ["negative", "neutral", "positive"]
                logging.info(f"Sentiment analysis: {sentiment_labels[predicted_sentiment]}")

            context_model.update_context(prompt, response_text, None)
            dynamic_planner.update_plan(prompt, response_text, None, 0.0, 0.0)  # Placeholder values for relevance and insightfulness scores
            return response_text

        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return ""
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return ""

def guess_language(code_snippet: str) -> str:
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
    code_pattern = re.compile(r'```(.*?)\n(.*?)\n```', re.DOTALL)
    return code_pattern.findall(text)

def simplify_prompt(prompt: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:30]

def main():
    if len(sys.argv) < 2:
        logging.error("No prompt provided. Usage: python script.py 'Your prompt here'")
        sys.exit(1)

    user_prompt = ' '.join(sys.argv[1:])
    try:
        chatgpt_worker = ChatGPTWorker()
        context_model = ContextAwareModel()
        dynamic_planner = DynamicPlanner()
        role = "Assistant"  # Set the desired role for ChatGPT
        response_text = chatgpt_worker.generate_response(user_prompt, role, context_model, dynamic_planner)
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