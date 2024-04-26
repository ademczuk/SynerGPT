# data_handling.py
import random
import logging
from typing import List, Dict
from pymongo import MongoClient

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB connection details
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
MONGODB_DATABASE = 'conversation_db'
MONGODB_COLLECTION = 'dataset'

# Connect to MongoDB
mongo_client = MongoClient(MONGODB_HOST, MONGODB_PORT)
db = mongo_client[MONGODB_DATABASE]
dataset_collection = db[MONGODB_COLLECTION]

def load_dataset() -> List[Dict[str, any]]:
    """
    Load the labeled dataset from MongoDB.
    The dataset format is as follows:
    [
        {
            "text": str,
            "feedback": str,
            "context": dict,
            "responder_name": str
        },
        ...
    ]
    Returns:
        List[Dict[str, any]]: A list of dictionaries containing the text, feedback, context, and responder name data.
    """
    try:
        dataset = list(dataset_collection.find())
        
        # Filter and transform the dataset to ensure the expected structure
        labeled_data = []
        for data in dataset:
            if 'text' in data and 'feedback' in data:
                labeled_data.append({
                    'text': data['text'],
                    'feedback': data['feedback'],
                    'context': data.get('context', {}),
                    'responder_name': data.get('responder_name', '')
                })
            else:
                logging.warning(f"Skipping data item due to missing 'text' or 'feedback' field: {data}")
        
        return labeled_data
    except Exception as e:
        logging.error(f"Error loading dataset from MongoDB: {str(e)}")
        return []

def generate_synthetic_data(num_samples: int = 100) -> List[Dict[str, any]]:
    """
    Generate synthetic data for training if the labeled dataset is insufficient.

    Args:
        num_samples (int): Number of synthetic data samples to generate.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing the synthetic text, feedback, context, and responder name data.
    """
    synthetic_data = []
    prompts = [
        "Write a short story about a brave adventurer.",
        "Describe your dream vacation destination.",
        "Explain the process of photosynthesis.",
        "Give advice on how to stay motivated and productive.",
        "Discuss the ethical implications of artificial intelligence."
    ]

    for _ in range(num_samples):
        prompt = random.choice(prompts)
        response = f"Synthetic response for prompt: {prompt}"
        feedback = "Sentiment: Neutral, Relevance: 0.8, Insightfulness: 0.6"
        context = {
            "topic": "Random topic",
            "sentiment": random.choice(["positive", "negative", "neutral"])
        }
        responder_name = random.choice(["ClaudeManager", "ChatGPTWorker"])

        synthetic_data.append({
            "text": response,
            "feedback": feedback,
            "context": context,
            "responder_name": responder_name
        })

    return synthetic_data

def save_dataset(dataset: List[Dict[str, any]]) -> None:
    """
    Save the dataset to MongoDB.
    The dataset format is as follows:
    [
        {
            "text": str,
            "feedback": str,
            "context": dict,
            "responder_name": str
        },
        ...
    ]
    Args:
        dataset (List[Dict[str, any]]): The dataset to be saved.
    """
    try:
        # Validate the dataset structure
        for data in dataset:
            if not isinstance(data, dict) or \
                    'text' not in data or \
                    'feedback' not in data or \
                    'context' not in data or \
                    'responder_name' not in data:
                raise ValueError(f"Invalid data structure: {data}")

        dataset_collection.delete_many({})  # Clear existing data
        dataset_collection.insert_many(dataset)
        logging.info("Dataset saved to MongoDB.")
    except Exception as e:
        logging.error(f"Error saving dataset to MongoDB: {str(e)}")

def clear_mongodb_dataset() -> None:
    """
    Clear the dataset collection in MongoDB.
    """
    try:
        dataset_collection.delete_many({})
        logging.info("MongoDB dataset cleared.")
    except Exception as e:
        logging.error(f"Error clearing MongoDB dataset: {str(e)}")

def ensure_minimum_dataset():
    try:
        dataset = load_dataset()
        if len(dataset) < 10:
            logging.warning("Labeled dataset has less than 10 samples. Generating synthetic data.")
            synthetic_data = generate_synthetic_data(num_samples=10 - len(dataset))
            dataset.extend(synthetic_data)
            save_dataset(dataset)
    except Exception as e:
        logging.error(f"Error ensuring minimum dataset: {str(e)}")