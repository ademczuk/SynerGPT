# AI-Driven Conversation System

This project implements an AI-driven conversation system that facilitates interactive and dynamic conversations between two language models, Claude (Manager) and ChatGPT (Worker). The system aims to achieve coherent, relevant, and insightful responses while adapting to the user's input and the conversation context.

Here's a summary of the project structure and code:
Project Structure:

The project has multiple Python files organized into a main directory and a subdirectory named "config".
The main directory contains various Python scripts for different functionalities such as data handling, modeling, evaluation, and utilities.
The "config" subdirectory contains a configuration file named "constants.py".

Code Overview:

analyze_response.py:

Contains functions for analyzing the sentiment, coherence, and relevance of a response.
Uses NLTK for tokenization and stopword removal, and gensim for Word2Vec modeling and similarity calculation.
Performs question answering using a pre-trained model.


ANtalk.py and Optalk.py:

Scripts for interacting with the Anthropic API and OpenAI API respectively.
Generate responses based on prompts and handle code extraction and saving.


ChatGPTWorker.py:

Defines the ChatGPTWorker class for generating responses using OpenAI's GPT-3 model.
Handles model loading, response generation, and sentiment analysis.


ClaudeManager.py:

Defines the ClaudeManager class for managing the interaction with the Claude model.
Handles model loading, prompt generation, and sentiment analysis.


context_aware_modeling.py:

Implements the ContextAwareModel class for context-aware modeling.
Provides methods for updating context, extracting sentiment, and retrieving relevant conversation history.


control.py:

The main control script for the AI-driven conversation system.
Handles model fine-tuning, data handling, and the main interaction loop between Claude and ChatGPT.
Utilizes reinforcement learning with the DynamicPlanner class.


data_handling.py:

Contains functions for loading and saving the labeled dataset using MongoDB.
Provides methods for generating synthetic data and ensuring a minimum dataset size.


dataset.py:

Defines the LabeledDataset class for handling labeled text data.
Implements methods for data loading, preprocessing, and batch generation.


evaluation.py:

Contains functions for evaluating the factuality, reasoning, coherence, and relevance of responses.
Utilizes external APIs for fact-checking and reasoning evaluation.


feedback_utils.py:

Provides functions for generating qualitative feedback based on relevance, insightfulness, factuality, and reasoning scores.
Handles saving feedback to the labeled dataset.


reinforcement_learning.py:

Implements the DynamicPlanner class for reinforcement learning-based planning.
Defines the Deep Q-Network (DQN) architecture and replay buffer.
Provides methods for selecting actions, updating the model, and adapting prompts.


semantic_similarity.py:

Contains functions for calculating semantic similarity between texts using SentenceTransformer.
Preprocesses text by tokenizing, lowercasing, and removing stopwords and punctuation.


sentiment_analysis.py and sentiment_module.py:

Provide functionality for sentiment analysis using NLTK's SentimentIntensityAnalyzer and a pre-trained BERT model.


topic_modeling.py:

Implements topic modeling using Latent Dirichlet Allocation (LDA) from the gensim library.
Extracts topics from text and generates topic clusters from conversation history.


train_model.py:

The main script for fine-tuning the BERT model for sequence classification.
Handles data loading, preprocessing, and model training using the Hugging Face Transformers library.
Utilizes Optuna for hyperparameter tuning.


utils.py:

Contains utility functions for text preprocessing, keyword extraction, sequence padding, one-hot encoding, and evaluation metrics.



The project also includes test files for unit testing and integration testing of various components.
Overall, the project implements an AI-driven conversation system that utilizes reinforcement learning, context-aware modeling, sentiment analysis, topic modeling, and fine-tuned language models to generate relevant and insightful responses. The system is designed to handle user prompts, maintain coherence in the conversation, and adapt based on feedback and context.

## Features

- Chained conversation between Claude (Manager) and ChatGPT (Worker)
- Context-aware modeling to maintain conversation coherence
- Reinforcement learning-based dynamic planning for adaptive response generation
- Sentiment analysis and topic modeling for enhanced conversation understanding
- Keyword extraction and synonym expansion for refined prompts
- Fine-tuning of BERT model for sentiment classification and relevance/insightfulness scoring
- Hyperparameter tuning using Optuna for optimal model performance
- Cross-validation and model evaluation metrics
- Integration with MLflow for experiment tracking and model versioning
- Logging and error handling for improved debugging and monitoring
- Unit tests for key functionalities

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- NLTK
- spaCy
- scikit-learn
- SentenceTransformers
- Optuna
- MLflow
- Flask
- MongoDB

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-driven-conversation-system.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ANTHROPIC_API_KEY`: Your Anthropic API key

4. Download the required spaCy and NLTK models:
   ```
   python -m spacy download en_core_web_sm
   python -m nltk.downloader wordnet
   ```

5. Set up the MongoDB database and update the connection details in `data_handling.py`.

## Usage

1. Prepare your labeled dataset in the required format and store it in MongoDB.

2. Fine-tune the BERT model:
   ```
   python train_model.py
   ```

3. Start the conversation system:
   ```
   python main.py
   ```

4. Interact with the system by providing prompts and observing the generated responses.

## API Endpoints

- `/api/interact` (POST): Accepts a JSON payload with `prompt` and `max_cycles` fields. Initiates the conversation between Claude and ChatGPT based on the provided prompt and maximum conversation cycles.

## Project Structure

- `main.py`: Entry point of the conversation system
- `control.py`: Core logic for controlling the conversation flow
- `ClaudeManager.py`: Manager class for Claude
- `ChatGPTWorker.py`: Worker class for ChatGPT
- `train_model.py`: Script for fine-tuning the BERT model
- `dataset.py`: Custom dataset class for labeled data
- `data_handling.py`: Data loading, preprocessing, and storage utilities
- `evaluation.py`: Evaluation metrics and functions
- `feedback_utils.py`: Utilities for generating and processing feedback
- `context_aware_modeling.py`: Context-aware modeling for conversation coherence
- `reinforcement_learning.py`: Reinforcement learning-based dynamic planning
- `semantic_similarity.py`: Semantic similarity calculations
- `sentiment_analysis.py`: Sentiment analysis using NLTK
- `topic_modeling.py`: Topic modeling using Latent Dirichlet Allocation (LDA)
- `keyword_extraction.py`: Keyword extraction using spaCy
- `refine_prompt.py`: Prompt refinement and keyword expansion
- `config.py`: Configuration settings
- `requirements.txt`: List of required dependencies

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
