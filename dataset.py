import torch
from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    def __init__(self, texts, sentiments, relevance_scores, insightfulness_scores, tokenizer):
        self.texts = texts
        self.sentiments = sentiments
        self.relevance_scores = relevance_scores
        self.insightfulness_scores = insightfulness_scores
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        relevance = self.relevance_scores[idx]
        insightfulness = self.insightfulness_scores[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        encoding['labels'] = torch.tensor([sentiment, relevance, insightfulness], dtype=torch.float)
        return {key: value.squeeze(0) for key, value in encoding.items()}