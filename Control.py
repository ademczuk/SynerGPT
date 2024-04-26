# control.py
import os
import subprocess
import sys
import json
import logging
import warnings
import random
import torch
from flask import Flask, jsonify, request
from datetime import datetime
from typing import Tuple, Callable
import sqlite3
import cProfile
import io
import pstats
import shutil
import tempfile
from threading import Thread
from refine_prompt import refine_prompt
from ClaudeManager import ClaudeManager
from ChatGPTWorker import ChatGPTWorker
from feedback_utils import get_llm_feedback, save_to_labeled_dataset
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from data_handling import ensure_minimum_dataset, clear_mongodb_dataset
from transformers import BertConfig, BertForSequenceClassification
from accelerate import Accelerator, DataLoaderConfiguration
from train_model import fine_tune_model
import argparse

# Add the project's root directory to the Python module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['PYTHONWARNINGS'] = 'ignore:__main__'
warnings.filterwarnings("ignore", message="Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead:")
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataloader_config = DataLoaderConfiguration(
    dispatch_batches=None,
    split_batches=False,
    even_batches=True,
    use_seedable_sampler=True
)
accelerator = Accelerator(
    dataloader_config=dataloader_config
)

app = Flask(__name__)

os.environ['PYTHONWARNINGS'] = 'ignore:__main__'
warnings.filterwarnings("ignore", message="Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead:")

fine_tune_script_path = 'train_model.py'
model_finetuned_path = './model_finetuned/'  # Define model_finetuned_path variable

INITIATION_PROMPT = """
This is a CHAINED conversation between two LLMs, Claude (MANAGER) and ChatGPT (WORKER). Each RESPONSE from one LLM is used as the PROMPT for the next LLM, with a focus on maintaining responses within 1000 characters.
Rules:
1. Responses are logged with a timestamp in chat_data.json.
2. Claude, as the "MANAGER", directs the conversation towards achieving the user's task by:
   a. Breaking down the task into smaller, manageable subtasks.
   b. Providing clear instructions and context to ChatGPT.
   c. Encouraging ChatGPT to explore multiple perspectives and provide detailed insights.
   d. Reviewing ChatGPT's responses and offering constructive feedback.
   e. Ensuring that the conversation stays focused on the original user prompt.
3. ChatGPT, as the "WORKER", generates responses based on Claude's prompts by:
   a. Carefully analyzing the given subtask and instructions.
   b. Providing detailed, well-structured responses that address the task requirements.
   c. Offering unique insights and considering different angles when exploring the topic.
   d. Seeking clarification from Claude when needed to ensure accurate and relevant responses.
   e. Staying focused on the original user prompt and avoiding tangential discussions.
4. Each response should be relevant to the task at hand and aim to progressively build towards the overall goal.
5. Claude and ChatGPT should maintain a professional and collaborative tone throughout the conversation.
6. The primary goal for both LLMs is to achieve consistent relevance and insightfulness scores of 5 in every response cycle.
7. Higher relevance and insightfulness scores will unlock new levels of complexity and advanced conversational abilities.
8. Achievements and milestones will be recognized and rewarded throughout the conversation.
User's Task: """

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_model_state():
    """
    Checks the state of the model to determine if reinitialization is needed.

    Returns:
        str: 'needs_reinitialization' if the model needs to be reinitialized,
             or 'exists' otherwise.
    """
    if not os.path.exists(model_finetuned_path) or not os.listdir(model_finetuned_path):
        return 'needs_reinitialization'
    else:
        return 'exists'  

def determine_num_labels():
    """
    Determines the appropriate number of labels for the output classifier.

    Returns:
        int: The number of labels.
    """
    # Implement your logic to determine the number of labels. This could be:
    #   * Hardcoded (e.g., return 3)
    #   * Read from a configuration file
    #   * Inferred from the training data 
    # For now, let's hardcode it
    return 3

def reinitialize_model(model_path, num_labels):
  """
  Reinitialize the BERT model with a new configuration and save it to the specified path.

  Args:
      model_path (str): Path to the BERT model directory.
      num_labels (int): Number of labels for the output classifier layer.
  """
  print(f"Reinitializing model at {model_path} with {num_labels} labels.")

  try:
      # Load fine-tuned configuration or a fresh configuration
      if os.path.exists(os.path.join(model_path, 'config.json')):
          config = BertConfig.from_pretrained(model_path)
      else:
          config = BertConfig(num_labels=num_labels)  # Initialize fresh config

      model = BertForSequenceClassification(config)  

      # Save the reinitialized model 
      model.save_pretrained(model_path)
      print(f"Model reinitialized and saved to {model_path}")

      # Notify the rest of the system that the model has been updated
      # This might involve sending signals, updating variables, etc. (implementation specific)

  except Exception as e:
      logging.error(f"Error reinitializing model: {str(e)}")

@app.route('/api/interact', methods=['POST'])
def interact():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_cycles = data.get('max_cycles', 10)
        if not prompt:
                return jsonify({'error': 'Prompt is required.'}), 400
        try:
            max_cycles = int(max_cycles)
            if max_cycles <= 0:
                return jsonify({'error': 'Max cycles must be a positive integer.'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Max cycles must be a valid integer.'}), 400
        main(prompt, max_cycles)  # Call the main function directly
        
        response = {
            'message': 'Interaction completed successfully.'
        }
        return jsonify(response), 200
    except Exception as e:
        logging.error(f"An error occurred during interaction: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
    
def create_model_finetuned_directory(model_finetuned_path: str):
    print(f"Creating or verifying model finetuned directory at {model_finetuned_path}")
    if not os.path.exists(model_finetuned_path):
        os.makedirs(model_finetuned_path)
        logging.info(f"Created directory: {model_finetuned_path}")
    
    # Create config.json
    config = BertConfig()
    config.save_pretrained(model_finetuned_path)
    logging.info(f"Saved config.json to {model_finetuned_path}")
    
    # Create dummy tokenizer.json
    dummy_tokenizer = {
        "model": {
            "vocab_size": 30522,
            "pad_token_id": 0,
            "type_vocab_size": 2
        }
    }
    
    with open(os.path.join(model_finetuned_path, 'tokenizer.json'), 'w') as tokenizer_file:
        json.dump(dummy_tokenizer, tokenizer_file)
    
    logging.info(f"Saved dummy tokenizer.json to {model_finetuned_path}")
    
    # Create dummy pytorch_model.bin
    with open(os.path.join(model_finetuned_path, 'pytorch_model.bin'), 'wb') as model_file:
        model_file.write(b'dummy_model_data')
    
    logging.info(f"Saved dummy pytorch_model.bin to {model_finetuned_path}")

def run_flask():
    app.run(debug=True, use_reloader=False, host='0.0.0.0')

def get_initial_prompt_and_max_cycles() -> Tuple[str, int]:
    print("Retrieving initial prompt and max cycles.")
    initial_prompt = None
    max_cycles = 10  # Default value for max_cycles
    
    if len(sys.argv) >= 3:
        initial_prompt = ' '.join(sys.argv[1:-1])
        
        try:
            max_cycles = int(sys.argv[-1])
        except ValueError:
            logging.warning("Error: max_cycles must be an integer. Using default of 10.")
    
    if not initial_prompt:
        try:
            print("Please enter your prompt (press Enter twice to finish): ")
            lines = []
            
            while True:
                line = input()
                if line == '':
                    break
                lines.append(line)
            
            initial_prompt = '\n'.join(lines)
            
            if not initial_prompt.strip():
                print("Prompt cannot be empty. Using default prompt.")
                initial_prompt = "Please provide a detailed prompt for the conversation."
        except (KeyboardInterrupt, EOFError):
            print("\nKeyboard interruption detected. Exiting...")
            sys.exit(1)
    
    return initial_prompt, max_cycles

def save_initial_data():
    print("Saving initial data to chat_data.json.")
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

def save_to_database(data: dict):
    print(f"Saving interaction data to database for PromptID {data['PromptID']}.")
    try:
        conn = sqlite3.connect('conversation_history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversation
                     (Timestamp TEXT, PromptID INTEGER, Role TEXT, PromptText TEXT, ResponseText TEXT,
                      FeedbackScore TEXT, FollowUpPrompt TEXT, ResponderName TEXT)''')
        c.execute("INSERT INTO conversation VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (data['Timestamp'], data['PromptID'], data['Role'], data['PromptText'], data['ResponseText'],
                   data['FeedbackScore'], data['FollowUpPrompt'], data['ResponderName']))
        conn.commit()
        conn.close()
        
        try:
            with open('chat_data.json', 'r') as file:
                chat_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            chat_data = []
        
        chat_data.append(data)
        
        try:
            with open('chat_data.json', 'w') as file:
                json.dump(chat_data, file, indent=2)
        except Exception as e:
            logging.error(f"Error saving chat data to chat_data.json: {str(e)}")
    except sqlite3.Error as e:
        logging.error(f"Error saving data to database: {str(e)}")

def initialize_chat_data():
    print("Initializing chat_data.json.")
    try:
        with open('chat_data.json', 'r') as file:
            json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        with open('chat_data.json', 'w') as file:
            json.dump([], file)

def profile_code(func: Callable) -> Callable:
    """
    Decorator to profile the execution of a function and log the profiling results.
    Args:
        func (Callable): The function to be profiled.
    Returns:
        Callable: The decorated function.
    """
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        logging.info(f"Profiling results for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper

def reinitialize_data():
    print("Reinitializing database and JSON files.")
    try:
        if os.path.exists('conversation_history.db'):
            conn = sqlite3.connect('conversation_history.db')
            c = conn.cursor()
            c.execute('DROP TABLE IF EXISTS conversation')
            conn.commit()
            conn.close()
            logging.info("Database reinitialized.")
        else:
            logging.warning("conversation_history.db does not exist.")
    except sqlite3.Error as e:
        logging.error(f"Error reinitializing database: {str(e)}")
    
    try:
        if os.path.exists('chat_data.json'):
            with open('chat_data.json', 'w') as file:
                json.dump([], file)
            logging.info("chat_data.json reinitialized.")
        else:
            logging.warning("chat_data.json does not exist.")
    except Exception as e:
        logging.error(f"Error reinitializing chat_data.json: {str(e)}")
    
    try:
        if os.path.exists('labeled_dataset.json'):
            with open('labeled_dataset.json', 'w') as file:
                json.dump([], file)
            logging.info("labeled_dataset.json reinitialized.")
        else:
            logging.warning("labeled_dataset.json does not exist.")
    except Exception as e:
        logging.error(f"Error reinitializing labeled_dataset.json: {str(e)}")
    
    # Clear the MongoDB dataset
    clear_mongodb_dataset()

@profile_code
def main(prompt, max_cycles):
    """
    Main function to handle model operations based on command-line arguments.
    """
    try:
        # Model state handling
        model_state = check_model_state()
        if model_state == 'needs_reinitialization':
            num_labels = determine_num_labels() 
            reinitialize_model(model_finetuned_path, num_labels)
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        print("Temporary directory created for processing.")
        # Create the model_finetuned directory and its required files
        create_model_finetuned_directory(model_finetuned_path)
        
        # Fine-tune the model if necessary
        fine_tune_script_path = 'train_model.py'  # Path to the fine-tuning script
        if not fine_tune_model(fine_tune_script_path, model_finetuned_path):
            print("Fine-tuning process failed. Exiting...")
            return
        
        ensure_minimum_dataset()  # Ensure the labeled_dataset.json file has minimum data
        logging.info("Starting the main interaction process...")
        dynamic_planner = DynamicPlanner(state_dim=100, action_dim=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Create an instance of DynamicPlanner with the required arguments
        claude_manager = ClaudeManager(dynamic_planner)  # Pass the dynamic_planner instance to ClaudeManag           
        chatgpt_worker = ChatGPTWorker()

        save_initial_data()
        original_prompt = prompt  # Store the original user prompt

        if not prompt or max_cycles is None:
            print("Error: Invalid initial prompt or max_cycles value.")
            return

        chat_log = f"{INITIATION_PROMPT}\n\nUser Prompt: {prompt}"
        conversation_history = []
        context_model = ContextAwareModel()  # Create an instance of ContextAwareModel
        dynamic_planner.conversation_history = []
        previous_scores = {"Claude": (0.0, 0.0), "ChatGPT": (0.0, 0.0)}
        response = ""  # Initialize the response variable

        for cycle_count in range(max_cycles):
            if cycle_count % 2 == 0:
                role = claude_manager.assign_role()
                refined_prompt = refine_prompt(response, "Claude", conversation_history[-1:], context_model, original_prompt, claude_manager.model, claude_manager.tokenizer, claude_manager.device, dynamic_planner)
                
                if claude_manager.is_model_loaded:
                    try:
                        response = claude_manager.generate_prompt(previous_response=response, conversation_history=conversation_history, context_model=context_model, original_prompt=original_prompt)
                    except Exception as e:
                        logging.error(f"Error generating prompt: {str(e)}")    
                else:
                    response = "Please provide more details on the topic."
                
                chatgpt_worker_response = conversation_history[-1]['response'] if conversation_history else ""
                responder_name = "ClaudeManager"
            else:
                role = "Assistant"
                refined_prompt = refine_prompt(response, "ChatGPT", conversation_history[-1:], context_model, original_prompt, claude_manager.model, claude_manager.tokenizer, claude_manager.device, dynamic_planner)
                response = chatgpt_worker.generate_response(refined_prompt, role, context_model, dynamic_planner)
                claude_manager_response = conversation_history[-1]['response'] if conversation_history else ""
                responder_name = "ChatGPTWorker"
            
            llm_name = "Claude" if cycle_count % 2 == 0 else "ChatGPT"
            previous_relevance, previous_insightfulness = previous_scores[llm_name]
            predicted_sentiment, relevance_score, insightfulness_score = get_llm_feedback(response, llm_name, (previous_relevance, previous_insightfulness))
            previous_scores[llm_name] = (relevance_score, insightfulness_score)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"\n{timestamp}\nCycle #: {cycle_count}\nRole: {role}\nResponder: {responder_name}\nPrompt: {refined_prompt}\nResponse: {response}\n"
            print(log_entry)
            logging.info(log_entry.replace('\\\\', '\\'))
            
            save_to_database({
                "Timestamp": timestamp,
                "PromptID": cycle_count,
                "Role": role,
                "PromptText": refined_prompt,
                "ResponseText": response,
                "FeedbackScore": f"Sentiment: {predicted_sentiment}, Relevance: {relevance_score:.2f}, Insightfulness: {insightfulness_score:.2f}",
                "FollowUpPrompt": "",
                "ResponderName": responder_name
            })
            
            conversation_history.append({"prompt": refined_prompt, "response": response})
            chat_log += f"\n{response}\n"
            context_model.update_context(refined_prompt, response, None)
            state = dynamic_planner.get_state(refined_prompt, response)
            action = dynamic_planner.select_action(state)
            reward = dynamic_planner.get_reward(refined_prompt, response)
            next_state = dynamic_planner.get_state(refined_prompt, response)
            done = 1 if reward >= 4.5 else 0
            dynamic_planner.update_model(state, action, reward, next_state, done)
            dynamic_planner.update_plan(refined_prompt, response, None, relevance_score, insightfulness_score)
            
            if cycle_count % 2 == 0:
                save_to_labeled_dataset(chatgpt_worker_response, "", context_model, responder_name, dynamic_planner)
            else:
                save_to_labeled_dataset(claude_manager_response, "", context_model, responder_name, dynamic_planner)  # Pass dynamic_planner
            
            # Introduce randomness in task selection and scoring criteria
            if random.random() < 0.2:
                # Modify the scoring criteria or task complexity
                # Implement your logic here to adjust the scoring criteria or task complexity
                pass
            
            # Monitor for patterns that suggest gaming of the system
            # Implement your logic here to analyze the conversation history and responses for potential gaming patterns
            # Adjust the scoring mechanisms and algorithms accordingly
            
            # Save conversation history to chat_data.json
            try:
                with open('chat_data.json', 'w') as file:
                    json.dump(conversation_history, file, indent=2)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)  # Clean up the temporary directory
            print("Cleaned up the temporary directory.")
            logging.info("Cleaned up the temporary directory.")

        # Continue execution after cleaning up the temporary directory
        print("Continuing execution after cleaning up the temporary directory.")

# Start the Flask server when the script is run directly
if __name__ == '__main__':
    # Initial setup: Ensure model directory is created and model is fine-tuned before starting the server.
    fine_tune_script_path = 'train_model.py'
    model_finetuned_path = './model_finetuned/' 
    create_model_finetuned_directory(model_finetuned_path)
    fine_tune_model(fine_tune_script_path, model_finetuned_path)
    # Argument parsing
    parser = argparse.ArgumentParser(description="Control script for handling BERT model operations.")
    parser.add_argument("-r", "--reinitialize", action="store_true",
                    help="Reinitialize the BERT model with a specified number of output labels.")
    parser.add_argument("--model_path", type=str, default=model_finetuned_path,
                    help="Path to the BERT model directory.")
    parser.add_argument("--num_labels", type=int, default=2,
                    help="Number of labels for the output classifier layer of the BERT model.")

    args = parser.parse_args()

    # Handle model operations based on command-line arguments
    if args.reinitialize:
        reinitialize_model(args.model_path, args.num_labels)
    else:
        logging.info("Starting the Flask server...")
        app.run(debug=True, host='0.0.0.0')
        # Starting the Flask server in a separate thread.
        flask_thread = Thread(target=run_flask)
        flask_thread.start()
        # Optionally wait for the Flask server thread to finish, although typically the server runs indefinitely.
        flask_thread.join()  # This line is optional and usually omitted in production servers.

