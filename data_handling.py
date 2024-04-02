import os
import json
import random
import sqlite3
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_path: str = 'labeled_dataset.db') -> List[Dict[str, any]]:
    """
    Load the labeled dataset from an SQLite database file.

    Args:
        dataset_path (str): Path to the SQLite database file.

    Returns:
        List[Dict[str, any]]: A list of dictionaries containing the text, feedback, context, and responder name data.
    """
    # Use an absolute path for the database file
    dataset_path = os.path.abspath(dataset_path)

    # Check if the database file exists
    if not os.path.isfile(dataset_path):
        logging.warning(f"Database file '{dataset_path}' does not exist. Creating a new file.")

    dataset = []
    try:
        with sqlite3.connect(dataset_path) as conn:
            c = conn.cursor()

            # Create the "dataset" table if it doesn't exist
            c.execute('''CREATE TABLE IF NOT EXISTS dataset
                         (text TEXT, feedback TEXT, context TEXT, responder_name TEXT)''')

            c.execute('SELECT * FROM dataset')
            rows = c.fetchall()
            for row in rows:
                text, feedback, context_json, responder_name = row
                dataset.append({
                    'text': text,
                    'feedback': feedback,
                    'context': json.loads(context_json),
                    'responder_name': responder_name
                })
    except sqlite3.Error as e:
        logging.error(f"Error loading dataset from SQLite: {str(e)}")

    return dataset

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

def save_dataset(dataset: List[Dict[str, any]], dataset_path: str = 'labeled_dataset.db') -> None:
    """
    Save the dataset (either loaded or synthetic) to an SQLite database file.

    Args:
        dataset (List[Dict[str, any]]): The dataset to be saved.
        dataset_path (str): Path to the SQLite database file.
    """
    # Use an absolute path for the database file
    dataset_path = os.path.abspath(dataset_path)

    try:
        with sqlite3.connect(dataset_path) as conn:
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS dataset (text TEXT, feedback TEXT, context TEXT, responder_name TEXT)')
            for data in dataset:
                context_json = json.dumps(data['context'])
                c.execute("INSERT INTO dataset VALUES (?, ?, ?, ?)", (data['text'], data['feedback'], context_json, data['responder_name']))
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error saving dataset to SQLite: {str(e)}")

def ensure_minimum_dataset():
    try:
        with open('labeled_dataset.json', 'r') as file:
            labeled_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        labeled_data = []

    if not labeled_data:
        sample_data = [
            {
                "text": "Artificial intelligence has the potential to revolutionize various industries, from healthcare to transportation. However, it also raises ethical concerns regarding job displacement and privacy.",
                "feedback": "The response provides a balanced view of the potential benefits and risks associated with artificial intelligence. It touches upon key areas such as healthcare, transportation, job displacement, and privacy concerns.",
                "context": [
                    "artificial intelligence",
                    "industries",
                    "ethical concerns"
                ],
                "responder_name": "ClaudeManager",
                "dynamic_plan": [
                    "Discuss the potential applications of artificial intelligence in healthcare and transportation.",
                    "Explore the ethical implications of job displacement caused by AI automation.",
                    "Analyze the privacy concerns related to AI and data collection."
                ]
            }
        ]

        try:
            with open('labeled_dataset.json', 'w') as file:
                json.dump(sample_data, file, indent=2)
        except Exception as e:
            logging.error(f"Error saving minimum dataset: {str(e)}")