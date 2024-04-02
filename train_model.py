import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from sklearn.model_selection import train_test_split
import logging
import argparse
import os
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_metric
from data_handling import load_dataset, generate_synthetic_data, save_dataset
from dataset import LabeledDataset
import warnings
from typing import List, Tuple
from torch.optim import AdamW
from accelerate import Accelerator, DataLoaderConfiguration

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(dataset_path):
    try:
        # Load the labeled dataset
        labeled_data = load_dataset(dataset_path)

        # Check if the dataset has enough data for training
        min_examples = 5
        if len(labeled_data) < min_examples:
            logging.warning(f"Insufficient labeled data for training. Generating synthetic data. (Found {len(labeled_data)} examples, minimum required: {min_examples})")
            synthetic_data = generate_synthetic_data(num_samples=100)
            labeled_data.extend(synthetic_data)
            save_dataset(labeled_data, dataset_path)

        # Extract text, sentiment, relevance, and insightfulness from the labeled dataset
        texts = [data['text'] for data in labeled_data]
        sentiments = [label_to_int(data['feedback'].split(':')[1].strip().split(',')[0]) for data in labeled_data]
        relevance_scores = [float(data['feedback'].split(',')[1].strip().split(':')[1]) for data in labeled_data]
        insightfulness_scores = [float(data['feedback'].split(',')[2].strip().split(':')[1]) for data in labeled_data]

        # Split the dataset into training and validation sets
        train_texts, val_texts, train_sentiments, val_sentiments, train_relevance, val_relevance, train_insightfulness, val_insightfulness = train_test_split(
            texts, sentiments, relevance_scores, insightfulness_scores, test_size=0.2, random_state=2021
        )

        return train_texts, val_texts, train_sentiments, val_sentiments, train_relevance, val_relevance, train_insightfulness, val_insightfulness
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {str(e)}")
        raise e

def label_to_int(label: str) -> int:
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    return label_map.get(label, 1)

def train_model(train_dataset, val_dataset, tokenizer):
    try:
        # Load BERT model for Sequence Classification
        bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        bert_model.to(device)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            use_cpu=True if not torch.cuda.is_available() else False,
        )

        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=None,
            split_batches=False,
            even_batches=True,
            use_seedable_sampler=True
        )
        accelerator = Accelerator(dataloader_config=dataloader_config)

        # Initialize Trainer
        trainer = Trainer(
            model=bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            optimizers=(AdamW(bert_model.parameters(), lr=5e-5), None),
        )

        # Check if a saved checkpoint exists and resume training from there
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logging.info(f"Resuming training from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            # Train the model
            trainer.train()
            logging.info("Model training completed successfully.")

        # Save the fine-tuned model
        model_save_path = './model_finetuned/'
        bert_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logging.info(f"Fine-tuned model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise e

def compute_metrics(eval_pred: Tuple[List[float], List[float]]):
    predictions, labels = eval_pred
    accuracy = load_metric("accuracy")
    result = accuracy.compute(predictions=predictions, references=labels, average="macro")
    return result

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a BERT model for sequence classification.')
    parser.add_argument('--dataset', type=str, default='labeled_dataset.db', help='Path to the labeled dataset file.')
    args = parser.parse_args()

    try:
        train_texts, val_texts, train_sentiments, val_sentiments, train_relevance, val_relevance, train_insightfulness, val_insightfulness = load_and_preprocess_data(args.dataset)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train_dataset = LabeledDataset(train_texts, train_sentiments, train_relevance, train_insightfulness, tokenizer)
        val_dataset = LabeledDataset(val_texts, val_sentiments, val_relevance, val_insightfulness, tokenizer)

        train_model(train_dataset, val_dataset, tokenizer)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()