# utils.py
import numpy as np
import nltk
import pandas as pd
import string
import spacy
import logging
from typing import List, Dict, Tuple, NamedTuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from update_model_data import UpdateModelData, get_update_model_data

class get_sentiment(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords from the given text using spaCy.

    Args:
        text (str): The text to extract keywords from.
        num_keywords (int): The number of keywords to extract.

    Returns:
        List[str]: The list of extracted keywords.
    """
    try:
        # Process the text with spaCy
        doc = nlp(text)

        # Extract keywords using spaCy's built-in functionality
        keywords = [chunk.text.lower() for chunk in doc.noun_chunks if chunk.root.pos_ in ["NOUN", "PROPN"]]

        # Remove duplicates and sort by frequency
        keyword_counts = {keyword: keywords.count(keyword) for keyword in set(keywords)}
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

        # Return the top N keywords
        return [keyword for keyword, _ in sorted_keywords[:num_keywords]]
    except Exception as e:
        logging.error(f"Error in extract_keywords: {str(e)}")
        return []

# Example usage for accessing fields of the NamedTuple
def example_usage(data: UpdateModelData):
    """
    Example usage of the UpdateModelData NamedTuple.

    Args:
        data (UpdateModelData): The UpdateModelData instance.

    Example:
        >>> state = np.array([0.5, 0.2, 0.1])
        >>> action = 1
        >>> reward = 0.8
        >>> next_state = np.array([0.6, 0.3, 0.2])
        >>> done = False
        >>> data = UpdateModelData(state, action, reward, next_state, done)
        >>> example_usage(data)
        State: [0.5 0.2 0.1]
        Action: 1
        Reward: 0.8
    """
    print(f"State: {data.state}")
    print(f"Action: {data.action}")
    print(f"Reward: {data.reward}")
    # Add more print statements or processing logic here


# Text Preprocessing
def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text data by lowercasing, removing punctuation, and tokenizing.

    Args:
        text (str): The input text to preprocess.

    Returns:
        List[str]: The preprocessed tokens.

    Example:
        >>> text = "Hello, World! This is a sample text."
        >>> preprocess_text(text)
        ['hello', 'world', 'this', 'is', 'a', 'sample', 'text']
    """
    if not text:
        return []
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    return tokens


# Sequence Padding
def pad_sequences(sequences: List[List[str]], max_length: int, padding_value: int = 0) -> List[List[int]]:
    """
    Pad sequences to the specified max length.

    Args:
        sequences (List[List[str]]): List of sequences to be padded.
        max_length (int): The desired maximum length of the padded sequences.
        padding_value (int): The value used for padding (default is 0).

    Returns:
        List[List[int]]: List of padded sequences.

    Raises:
        ValueError: If max_length is less than or equal to zero.

    Example:
        >>> sequences = [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h', 'i']]
        >>> pad_sequences(sequences, max_length=3)
        [['a', 'b', 'c'], ['d', 'e', 0], ['f', 'g', 'h']]
    """
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    padded_sequences = []
    for sequence in sequences:
        padded_sequence = sequence[:max_length] + [padding_value] * (max_length - len(sequence))
        padded_sequences.append(padded_sequence)
    return padded_sequences


# One-Hot Encoding
def one_hot_encode(labels: List[int], num_classes: int) -> np.ndarray:
    """
    One-hot encode a list of labels.

    Args:
        labels (List[int]): List of labels to be one-hot encoded.
        num_classes (int): The total number of classes.

    Returns:
        np.ndarray: The one-hot encoded labels.

    Raises:
        ValueError: If num_classes is less than the maximum label value in labels.

    Example:
        >>> labels = [0, 1, 2, 1, 0]
        >>> one_hot_encode(labels, num_classes=3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 1., 0.],
               [1., 0., 0.]])
    """
    if num_classes < max(labels) + 1:
        raise ValueError("num_classes must be greater than or equal to the maximum label value in labels.")

    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels


# Probability to Label Conversion
def probabilities_to_labels(probabilities: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to class labels by selecting the index with the highest probability.

    Args:
        probabilities (np.ndarray): The input probabilities.

    Returns:
        np.ndarray: The predicted class labels.

    Example:
        >>> probabilities = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        >>> probabilities_to_labels(probabilities)
        array([1, 0, 1])
    """
    return np.argmax(probabilities, axis=1)


# Data Loading and Saving
def load_data_from_csv(filepath: str, text_column: str, label_column: str) -> Tuple[List[str], List[int]]:
    """
    Load text and label data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        text_column (str): The name of the column containing the text data.
        label_column (str): The name of the column containing the label data.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing the loaded text data and labels.

    Example:
        >>> load_data_from_csv('data.csv', 'text', 'label')
        (['Sample text 1', 'Sample text 2'], [0, 1])
    """
    df = pd.read_csv(filepath)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    return texts, labels


# Data Splitting
def split_data(texts: List[str], labels: List[int], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Split the data into training and testing sets.

    Args:
        texts (List[str]): The input text data.
        labels (List[int]): The corresponding labels.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): The seed used by the random number generator (default is 42).

    Returns:
        Tuple[List[str], List[str], List[int], List[int]]: A tuple containing the train texts, test texts, train labels, and test labels.

    Example:
        >>> texts = ['Sample text 1', 'Sample text 2', 'Sample text 3', 'Sample text 4']
        >>> labels = [0, 1, 0, 1]
        >>> train_texts, test_texts, train_labels, test_labels = split_data(texts, labels)
        >>> len(train_texts), len(test_texts), len(train_labels), len(test_labels)
        (3, 1, 3, 1)
    """
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
    return train_texts, test_texts, train_labels, test_labels


# Evaluation Metrics
def calculate_accuracy(true_labels: List[int], predicted_labels: List[int]) -> float:
    """
    Calculate the accuracy of the predicted labels.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.

    Returns:
        float: The calculated accuracy.

    Example:
        >>> true_labels = [0, 1, 2, 1, 0]
        >>> predicted_labels = [0, 1, 1, 1, 0]
        >>> calculate_accuracy(true_labels, predicted_labels)
        0.8
    """
    return accuracy_score(true_labels, predicted_labels)


def calculate_precision(true_labels: List[int], predicted_labels: List[int], average: str = 'micro') -> float:
    """
    Calculate the precision of the predicted labels.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.
        average (str): The averaging method ('micro', 'macro', 'weighted', or 'binary') (default is 'micro').

    Returns:
        float: The calculated precision.

    Example:
        >>> true_labels = [0, 1, 2, 1, 0]
        >>> predicted_labels = [0, 1, 1, 1, 0]
        >>> calculate_precision(true_labels, predicted_labels)
        0.8
    """
    return precision_score(true_labels, predicted_labels, average=average)


def calculate_recall(true_labels: List[int], predicted_labels: List[int], average: str = 'micro') -> float:
    """
    Calculate the recall of the predicted labels.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.
        average (str): The averaging method ('micro', 'macro', 'weighted', or 'binary') (default is 'micro').

    Returns:
        float: The calculated recall.

    Example:
        >>> true_labels = [0, 1, 2, 1, 0]
        >>> predicted_labels = [0, 1, 1, 1, 0]
        >>> calculate_recall(true_labels, predicted_labels)
        0.8
    """
    return recall_score(true_labels, predicted_labels, average=average)


def calculate_f1_score(true_labels: List[int], predicted_labels: List[int], average: str = 'micro') -> float:
    """
    Calculate the F1 score of the predicted labels.

    Args:
        true_labels (List[int]): The true labels.
        predicted_labels (List[int]): The predicted labels.
        average (str): The averaging method ('micro', 'macro', 'weighted', or 'binary') (default is 'micro').

    Returns:
        float: The calculated F1 score.

    Example:
        >>> true_labels = [0, 1, 2, 1, 0]
        >>> predicted_labels = [0, 1, 1, 1, 0]
        >>> calculate_f1_score(true_labels, predicted_labels)
        0.8
    """
    return f1_score(true_labels, predicted_labels, average=average)


def calculate_auc_roc(true_labels: List[int], predicted_probabilities: List[float], multi_class: str = 'ovr') -> float:
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) score.

    Args:
        true_labels (List[int]): The true labels.
        predicted_probabilities (List[float]): The predicted probabilities.
        multi_class (str): The multi-class mode ('ovr' for one-vs-rest or 'ovo' for one-vs-one) (default is 'ovr').

    Returns:
        float: The calculated AUC-ROC score.

    Example:
        >>> true_labels = [0, 1, 2, 1, 0]
        >>> predicted_probabilities = [0.2, 0.8, 0.6, 0.7, 0.3]
        >>> calculate_auc_roc(true_labels, predicted_probabilities)
        0.75
    """
    return roc_auc_score(true_labels, predicted_probabilities, multi_class=multi_class)


# Update the __all__ variable
__all__ = [
    'UpdateModelData', 'example_usage', 'preprocess_text', 'pad_sequences', 'one_hot_encode', 'probabilities_to_labels',
    'load_data_from_csv', 'split_data', 'calculate_accuracy', 'calculate_precision', 'calculate_recall', 'calculate_f1_score', 'calculate_auc_roc'
]