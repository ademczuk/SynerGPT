# train_model.py
import os
import subprocess
import logging
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
import logging
import argparse
from data_handling import load_dataset, save_dataset, generate_synthetic_data
from dataset import LabeledDataset
import optuna
from sklearn.metrics import precision_recall_fscore_support

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune_model(fine_tune_script_path, model_finetuned_path):
    """
    Fine-tune the model using the provided fine-tuning script and save the fine-tuned model to the specified path.

    Args:
        fine_tune_script_path (str): The path to the fine-tuning script.
        model_finetuned_path (str): The path to save the fine-tuned model.

    Returns:
        bool: True if fine-tuning is successful, False otherwise.
    """
    print("Checking if fine-tuning is needed...")
    if os.path.exists(model_finetuned_path):
        logging.info("Fine-tuned model already exists. Skipping fine-tuning process.")
        return True

    logging.info("Starting fine-tuning process...")

    try:
        process = subprocess.Popen(['python', fine_tune_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logging.error(f"Error occurred during fine-tuning: {stderr.decode('utf-8')}")
            return False
        else:
            logging.info("Fine-tuning process completed successfully.")
            return True
    except Exception as e:
        logging.error(f"Error occurred during fine-tuning: {str(e)}")
        return False

def load_and_preprocess_data():
    try:
        logging.info("Loading labeled dataset from MongoDB...")
        labeled_data = load_dataset()

        logging.info(f"Loaded {len(labeled_data)} examples from the dataset.")

        if not labeled_data:
            logging.warning("No labeled data found in the dataset. Generating synthetic data...")
            labeled_data = generate_synthetic_data()  # Generate synthetic data if the dataset is empty
            logging.info(f"Generated {len(labeled_data)} synthetic examples.")
            save_dataset(labeled_data)  # Save the synthetic data to MongoDB
            logging.info("Synthetic data saved to MongoDB.")

        texts = []
        sentiments = []
        relevance_scores = []
        insightfulness_scores = []

        for data in labeled_data:
            if isinstance(data, dict) and 'text' in data and 'feedback' in data:
                texts.append(data['text'])
                sentiments.append(label_to_int(data['feedback'].split(':')[1].strip().split(',')[0]))
                relevance_scores.append(float(data['feedback'].split(',')[1].strip().split(':')[1]))
                insightfulness_scores.append(float(data['feedback'].split(',')[2].strip().split(':')[1]))
            else:
                logging.warning(f"Skipping data item due to missing 'text' or 'feedback' field: {data}")

        if not texts or not sentiments or not relevance_scores or not insightfulness_scores:
            logging.warning("Insufficient labeled data. One or more required fields are empty.")
            return [], [], [], [], [], [], [], []

        logging.info("Splitting the dataset into training and validation sets...")
        # Split the dataset into training and validation sets using stratified k-fold cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_indices, val_indices = next(kfold.split(texts, sentiments))

        train_texts = [texts[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        train_sentiments = [sentiments[i] for i in train_indices]
        val_sentiments = [sentiments[i] for i in val_indices]
        train_relevance = [relevance_scores[i] for i in train_indices]
        val_relevance = [relevance_scores[i] for i in val_indices]
        train_insightfulness = [insightfulness_scores[i] for i in train_indices]
        val_insightfulness = [insightfulness_scores[i] for i in val_indices]

        logging.info("Data loading and preprocessing completed.")
        return train_texts, val_texts, train_sentiments, val_sentiments, train_relevance, val_relevance, train_insightfulness, val_insightfulness
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {str(e)}")
        return [], [], [], [], [], [], [], []

def label_to_int(label):
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    return label_map.get(label, 1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division='warn')
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def objective(trial, train_dataset, val_dataset, tokenizer):
    # Hyperparameter tuning using Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16])
    warmup_steps = trial.suggest_int('warmup_steps', 0, 1000)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)

    # Load BERT model for Sequence Classification
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    bert_model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(AdamW(bert_model.parameters(), lr=learning_rate), None),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_metrics = trainer.evaluate()
    accuracy = eval_metrics['eval_accuracy']

    return accuracy

def train_model(train_dataset, val_dataset, tokenizer):
    try:
        if len(train_dataset) == 0:
            logging.warning("Train dataset is empty. Skipping model training.")
            return None

        # Hyperparameter tuning using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, tokenizer), n_trials=20)
        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")

        # Load the best model for sequence classification
        best_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, ignore_mismatched_sizes=True)
        best_model.to(device)

        # Define training arguments with best hyperparameters
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=best_params['num_train_epochs'],
            per_device_train_batch_size=best_params['per_device_train_batch_size'],
            per_device_eval_batch_size=best_params['per_device_train_batch_size'],
            warmup_steps=best_params['warmup_steps'],
            weight_decay=best_params['weight_decay'],
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        # Initialize Trainer
        trainer = Trainer(
            model=best_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            optimizers=(AdamW(best_model.parameters(), lr=best_params['learning_rate']), None),
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_metrics = trainer.evaluate()
        accuracy = eval_metrics['eval_accuracy']

        # Save the best-performing model
        model_save_path = './model_finetuned/'
        best_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logging.info(f"Best-performing model saved to {model_save_path}")

        return best_model

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise e

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a BERT model for sequence classification.')
    args = parser.parse_args()

    try:
        logging.info("Loading and preprocessing data...")
        train_texts, val_texts, train_sentiments, val_sentiments, train_relevance, val_relevance, train_insightfulness, val_insightfulness = load_and_preprocess_data()

        if not train_texts or not val_texts or not train_sentiments or not val_sentiments or not train_relevance or not val_relevance or not train_insightfulness or not val_insightfulness:
            logging.error("Insufficient data for training. Exiting...")
            return

        logging.info("Initializing tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        logging.info("Creating train and validation datasets...")
        train_dataset = LabeledDataset(train_texts, train_sentiments, train_relevance, train_insightfulness, tokenizer)
        val_dataset = LabeledDataset(val_texts, val_sentiments, val_relevance, val_insightfulness, tokenizer)

        logging.info("Starting model training...")
        trained_model = train_model(train_dataset, val_dataset, tokenizer)

        if trained_model is None:
            logging.warning("Model training was skipped due to empty train dataset.")
        else:
            logging.info("Model training completed.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()