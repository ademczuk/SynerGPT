from train_model import train_model
from dataset import LabeledDataset
from transformers import BertTokenizer
from data_handling import load_dataset
from evaluation import evaluate_conversation

def main():
    # Load and preprocess data
    labeled_data = load_dataset('labeled_dataset.db')
    texts = [data['text'] for data in labeled_data]
    sentiments = [data['feedback'].split(':')[1].strip().split(',')[0] for data in labeled_data]
    relevance_scores = [float(data['feedback'].split(',')[1].strip().split(':')[1]) for data in labeled_data]
    insightfulness_scores = [float(data['feedback'].split(',')[2].strip().split(':')[1]) for data in labeled_data]

    # Tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = LabeledDataset(texts, sentiments, relevance_scores, insightfulness_scores, tokenizer)
    val_dataset = LabeledDataset(texts[:100], sentiments[:100], relevance_scores[:100], insightfulness_scores[:100], tokenizer)  # Use a subset for validation

    # Train model
    model = train_model(train_dataset, val_dataset, tokenizer)

    # Evaluate conversation
    conversation_history = [
        {"prompt": "What is the meaning of life?", "response": "The meaning of life is subjective and varies for each individual."},
        {"prompt": "How can I find happiness?", "response": "Happiness can be found through various means such as practicing gratitude, cultivating relationships, and pursuing meaningful goals."}
    ]
    evaluate_conversation(conversation_history)

if __name__ == "__main__":
    main()