# Control.py
import os
import subprocess
import sys
from datetime import datetime
import logging
import json
import warnings
import random
from refine_prompt import refine_prompt
from ClaudeManager import ClaudeManager
from ChatGPTWorker import ChatGPTWorker
from feedback_utils import get_llm_feedback, save_to_labeled_dataset
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from data_handling import ensure_minimum_dataset

os.environ['PYTHONWARNINGS'] = 'ignore:__main__'
warnings.filterwarnings("ignore", message="Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead:")

fine_tune_script_path = 'train_model.py'
model_finetuned_path = './model_finetuned/'

INITIATION_PROMPT = """
This is a CHAINED conversation between two LLMs, Claude (MANAGER) and ChatGPT (WORKER). Each RESPONSE from one LLM is used as the PROMPT for the next LLM, with a focus on maintaining responses within 1000 characters.

Rules:
1. Responses are logged with a timestamp in chat_data.json.
2. Claude, as the "MANAGER", directs the conversation towards achieving the user's task by:
   a. Breaking down the task into smaller, manageable subtasks.
   b. Providing clear instructions and context to ChatGPT.
   c. Encouraging ChatGPT to explore multiple perspectives and provide detailed insights.
   d. Reviewing ChatGPT's responses and offering constructive feedback.
3. ChatGPT, as the "WORKER", generates responses based on Claude's prompts by:
   a. Carefully analyzing the given subtask and instructions.
   b. Providing detailed, well-structured responses that address the task requirements.
   c. Offering unique insights and considering different angles when exploring the topic.
   d. Seeking clarification from Claude when needed to ensure accurate and relevant responses.
4. Each response should be relevant to the task at hand and aim to progressively build towards the overall goal.
5. Claude and ChatGPT should maintain a professional and collaborative tone throughout the conversation.
6. The primary goal for both LLMs is to achieve consistent relevance and insightfulness scores of 5 in every response cycle.
7. Higher relevance and insightfulness scores will unlock new levels of complexity and advanced conversational abilities.
8. Achievements and milestones will be recognized and rewarded throughout the conversation.

User's Task:
"""

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model_finetuned_directory():
    if not os.path.exists(model_finetuned_path):
        os.makedirs(model_finetuned_path)
        # Create initial files if necessary
        # For example, you can create a dummy tokenizer file
        with open(os.path.join(model_finetuned_path, 'tokenizer.json'), 'w') as file:
            json.dump({}, file)
        # Add any other necessary files

def fine_tune_model():
    if os.path.exists(model_finetuned_path):
        logging.info("Fine-tuned model already exists. Skipping fine-tuning process.")
        return

    logging.info("Starting fine-tuning process...")
    try:
        process = subprocess.Popen(['python', fine_tune_script_path])
        process.wait(timeout=3600)  # Set a timeout of 1 hour (adjust as needed)
        if process.returncode == 0:
            logging.info("Fine-tuning completed.")
        else:
            logging.error(f"Fine-tuning failed with return code: {process.returncode}")
    except subprocess.TimeoutExpired:
        logging.error("Fine-tuning process timed out.")
        process.terminate()  # Terminate the process if it times out
    except subprocess.CalledProcessError as e:
        logging.error(f"Fine-tuning failed: {str(e)}")

def get_initial_prompt_and_max_cycles():
    initial_prompt = None
    max_cycles = 10  # Default value for max_cycles

    if len(sys.argv) >= 3:
        initial_prompt = ' '.join(sys.argv[1:-1])
        try:
            max_cycles = int(sys.argv[-1])
        except ValueError:
            logging.warning("Error: max_cycles must be an integer. Using default of 10.")

    while not initial_prompt:
        try:
            print("Please enter your prompt (press Enter twice to finish): ")
            lines = []
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            initial_prompt = '\n'.join(lines)

            if initial_prompt.strip():
                break
            else:
                print("Prompt cannot be empty. Please enter a valid prompt.")
        except (KeyboardInterrupt, EOFError):
            print("\nKeyboard interruption detected. Exiting...")
            sys.exit(1)

    return initial_prompt, max_cycles

def save_initial_data():
    if not os.path.exists('chat_data.json') or os.path.getsize('chat_data.json') == 0:
        initial_data = [
            {"Timestamp": "2023-06-10 10:00:00", "PromptID": 0, "Role": "User", "PromptText": "Initial prompt", "ResponseText": "Initial response", "FeedbackScore": "Relevance: 4, Insightfulness: 3", "FollowUpPrompt": "Follow-up prompt", "ResponderName": "ChatGPTWorker"},
            {"Timestamp": "2023-06-10 10:05:00", "PromptID": 1, "Role": "Assistant", "PromptText": "Follow-up prompt", "ResponseText": "Follow-up response", "FeedbackScore": "Relevance: 5, Insightfulness: 4", "FollowUpPrompt": "Another follow-up prompt", "ResponderName": "ClaudeManager"}
        ]
        
        try:
            with open('chat_data.json', 'w') as file:
                json.dump(initial_data, file, indent=4)
        except Exception as e:
            logging.error(f"Error saving initial data: {str(e)}")

def save_to_json(data):
    try:
        with open('chat_data.json', 'r') as file:
            chat_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Error loading chat data: {str(e)}")
        chat_data = []

    chat_data.append(data)

    try:
        with open('chat_data.json', 'w') as file:
            json.dump(chat_data, file, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving chat data: {str(e)}")

def main():
    create_model_finetuned_directory()
    fine_tune_model()
    ensure_minimum_dataset()  # Ensure the labeled_dataset.json file has minimum data
    logging.info("Starting the main interaction process...")

    claude_manager = ClaudeManager()
    chatgpt_worker = ChatGPTWorker()

    save_initial_data()
    
    initial_prompt, max_cycles = get_initial_prompt_and_max_cycles()
    if not initial_prompt or max_cycles is None:
        print("Error: Invalid initial prompt or max_cycles value.")
        return

    chat_log = f"{INITIATION_PROMPT}\n\nUser Prompt: {initial_prompt}"
    conversation_history = []
    
    context_model = ContextAwareModel()  # Create an instance of ContextAwareModel
    dynamic_planner = DynamicPlanner()  # Create an instance of DynamicPlanner
    
    previous_scores = {"Claude": (0.0, 0.0), "ChatGPT": (0.0, 0.0)}

    response = ""  # Initialize the response variable

    for cycle_count in range(max_cycles):
        if cycle_count % 2 == 0:
            role = claude_manager.assign_role()
            refined_prompt = refine_prompt(response, "Claude", conversation_history[-1:], context_model)  # Pass only the last conversation entry
            if claude_manager.is_model_loaded:
                response = claude_manager.generate_prompt(refined_prompt, conversation_history[-1:], context_model, dynamic_planner)  # Pass only the last conversation entry and dynamic_planner
            else:
                response = "Please provide more details on the topic."
            feedback_prompt = "Please provide feedback on the previous response from ChatGPT:\n"
            if conversation_history:
                feedback_prompt += f"{conversation_history[-1]['response']}\n\nRelevance (1-5): \nCoherence (1-5): \nInsightfulness (1-5): \n"
            if claude_manager.is_model_loaded:
                feedback = claude_manager.generate_prompt(feedback_prompt, conversation_history[-1:], context_model, dynamic_planner)  # Pass only the last conversation entry and dynamic_planner
            else:
                feedback = "No feedback available."
            chatgpt_worker_response = conversation_history[-1]['response'] if conversation_history else ""
            responder_name = "ClaudeManager"
        else:
            role = "Assistant"
            refined_prompt = refine_prompt(response, "ChatGPT", conversation_history[-1:], context_model)  # Pass only the last conversation entry
            response = chatgpt_worker.generate_response(refined_prompt, role, context_model, dynamic_planner)  # Pass dynamic_planner
            feedback_prompt = "Please provide feedback on the previous response from Claude:\n"
            if conversation_history:
                feedback_prompt += f"{conversation_history[-1]['response']}\n\nRelevance (1-5): \nCoherence (1-5): \nInsightfulness (1-5): \n"
            feedback = chatgpt_worker.generate_response(feedback_prompt, role, context_model, dynamic_planner)  # Pass dynamic_planner
            claude_manager_response = conversation_history[-1]['response'] if conversation_history else ""
            responder_name = "ChatGPTWorker"

        llm_name = "Claude" if cycle_count % 2 == 0 else "ChatGPT"
        previous_relevance, previous_insightfulness = previous_scores[llm_name]
        predicted_sentiment, relevance_score, insightfulness_score = get_llm_feedback(response, llm_name, (previous_relevance, previous_insightfulness))
        previous_scores[llm_name] = (relevance_score, insightfulness_score)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n{timestamp}\nCycle #: {cycle_count}\nRole: {role}\nResponder: {responder_name}\nPrompt: {refined_prompt}\nResponse: {response}\n"
        print(log_entry)  # Print the log entry to the terminal
        logging.info(log_entry.replace('\\\\', '\\'))  # Log the entry with escaped backslashes

        save_to_json({
            "Timestamp": timestamp,
            "PromptID": cycle_count,
            "Role": role,
            "PromptText": refined_prompt,
            "ResponseText": response,
            "FeedbackScore": f"Sentiment: {predicted_sentiment}, Relevance: {relevance_score:.2f}, Insightfulness: {insightfulness_score:.2f}",
            "FollowUpPrompt": "",
            "ResponderName": responder_name
        })
        
        conversation_history.append({"prompt": refined_prompt, "response": response, "feedback": feedback})
        chat_log += f"\n{response}\n\nFeedback: {feedback}"

        context_model.update_context(refined_prompt, response, feedback)  # Update the context model
        dynamic_planner.update_plan(refined_prompt, response, feedback, relevance_score, insightfulness_score)  # Update the dynamic planner

        # Save feedback to labeled dataset
        if cycle_count % 2 == 0:
            save_to_labeled_dataset(chatgpt_worker_response, feedback, context_model, responder_name, dynamic_planner)  # Pass dynamic_planner
        else:
            save_to_labeled_dataset(claude_manager_response, feedback, context_model, responder_name, dynamic_planner)  # Pass dynamic_planner

        # Introduce randomness in task selection and scoring criteria
        if random.random() < 0.2:
            # Modify the scoring criteria or task complexity
            # Implement your logic here to adjust the scoring criteria or task complexity
            pass

        # Monitor for patterns that suggest gaming of the system
        # Implement your logic here to analyze the conversation history and responses for potential gaming patterns
        # Adjust the scoring mechanisms and algorithms accordingly

if __name__ == '__main__':
    main()