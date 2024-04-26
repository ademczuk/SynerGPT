# main.py
from train_model import train_model
from dataset import LabeledDataset
from transformers import BertTokenizer
from data_handling import load_dataset
from evaluation import evaluate_conversation
import logging
import argparse
from flask import Flask, jsonify, request
from Control import main as control_main

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/api/interact', methods=['POST'])
def interact():
    """
    API endpoint for interaction.
    Accepts a JSON payload with 'prompt' and 'max_cycles' fields.
    Calls the main function from control.py with the provided prompt and max_cycles.
    Returns a JSON response indicating the completion status.
    """
    data = request.json
    prompt = data.get('prompt', '')
    max_cycles = data.get('max_cycles', 10)
    
    # Call the main function from control.py with the provided prompt and max_cycles
    control_main(prompt, max_cycles)
    
    response = {
        'message': 'Interaction completed successfully.'
    }
    return jsonify(response), 200

def main():
    """
    Main function for the AI-Driven Conversation System.
    Parses command-line arguments, loads and preprocesses data, trains the model,
    evaluates the conversation, and starts the Flask server.
    """
    parser = argparse.ArgumentParser(description='AI-Driven Conversation System')
    parser.add_argument('--dataset_path', type=str, default='labeled_dataset.db', help='Path to the labeled dataset file')
    parser.add_argument('--model_path', type=str, default='model_finetuned', help='Path to save the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        logging.info(f"Loading dataset from {args.dataset_path}")
        labeled_data = load_dataset(args.dataset_path)
        
        texts = [data['text'] for data in labeled_data]
        sentiments = [data['feedback'].split(':')[1].strip().split(',')[0] for data in labeled_data]
        relevance_scores = [float(data['feedback'].split(',')[1].strip().split(':')[1]) for data in labeled_data]
        insightfulness_scores = [float(data['feedback'].split(',')[2].strip().split(':')[1]) for data in labeled_data]
        
        logging.info("Tokenizing data")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = LabeledDataset(texts, sentiments, relevance_scores, insightfulness_scores, tokenizer)
        val_dataset = LabeledDataset(texts[-100:], sentiments[-100:], relevance_scores[-100:], insightfulness_scores[-100:], tokenizer)
        
        # Train model
        logging.info("Starting model training")
        model = train_model(train_dataset, val_dataset, tokenizer, args.model_path, args.num_epochs, args.batch_size, args.learning_rate)
        
        # Evaluate conversation
        logging.info("Evaluating conversation")
        conversation_history = [
            {"prompt": "What is the meaning of life?", "response": "The meaning of life is subjective and varies for each individual."},
            {"prompt": "How can I find happiness?", "response": "Happiness can be found through various means such as practicing gratitude, cultivating relationships, and pursuing meaningful goals."}
        ]
        # Assuming conversation_history is a list of dictionaries
        # with keys 'prompt' and 'response'
        evaluate_conversation(conversation_history)
        logging.info("Conversation evaluation completed")
        
        # Start the Flask server
        app.run(debug=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()