# dataset.py
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LabeledDataset(Dataset):
    def __init__(self, texts: List[str], sentiments: List[int], relevance_scores: List[float], insightfulness_scores: List[float], tokenizer):
        """
        Initialize the LabeledDataset.
        Args:
            texts (List[str]): List of text samples.
            sentiments (List[int]): List of sentiment labels.
            relevance_scores (List[float]): List of relevance scores.
            insightfulness_scores (List[float]): List of insightfulness scores.
            tokenizer: The tokenizer for text encoding.
        """
        self.texts = texts
        self.sentiments = sentiments
        self.relevance_scores = relevance_scores
        self.insightfulness_scores = insightfulness_scores
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        Returns:
            int: The number of samples.
        """
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        Args:
            index (int): The index of the sample.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the encoded text, sentiment label, relevance score, and insightfulness score.
        """
        text = self.texts[index]
        sentiment = self.sentiments[index]
        relevance = self.relevance_scores[index]
        insightfulness = self.insightfulness_scores[index]

        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentiment, dtype=torch.long),  # Add this line to include the labels
            'relevance': torch.tensor(relevance, dtype=torch.float),
            'insightfulness': torch.tensor(insightfulness, dtype=torch.float)
        }

    def get_labels(self) -> List[int]:
        """
        Get the list of sentiment labels for the dataset.
        Returns:
            List[int]: The list of sentiment labels.
        """
        return self.sentiments