# topic_modeling.py
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import logging
from gensim.models import LdaMulticore
from typing import List, Dict
import random

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lda_topic_modeling(text: str, num_topics: int = 3, num_words: int = 5) -> List[str]:
    try:
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        # Create dictionary and corpus
        dictionary = corpora.Dictionary([preprocessed_text])
        corpus = [dictionary.doc2bow(preprocessed_text)]
        # Train LDA model
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, workers=4)
        # Extract topics
        topics = []
        for topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
            topic_words = topic[1].split('+')
            topic_words = [word.split('*')[1].strip('"') for word in topic_words]
            topics.extend(topic_words)
        return topics
    except Exception as e:
        logging.error(f"Error in lda_topic_modeling: {str(e)}")
        return []

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess the text by tokenizing, lowercasing, removing stop words and punctuation.

    Args:
        text (str): The input text.

    Returns:
        List[str]: The preprocessed tokens.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return filtered_tokens

def evaluate_topic_coherence(topic_words: List[str]) -> float:
    """
    Evaluate the coherence of a topic based on its top words.
    Args:
        topic_words (List[str]): The top words representing the topic.
    Returns:
        float: The topic coherence score.
    """
    # Implement the topic coherence evaluation logic using measures like PMI or NPMI
    # Calculate the coherence score based on the co-occurrence of topic words in a reference corpus
    # Return the coherence score as a float value
    # Example implementation:
    coherence_score = random.random()  # Replace with your topic coherence evaluation logic
    return coherence_score

def extract_topics(text: str, num_topics: int = 3, num_words: int = 5) -> List[str]:
    """
    Extract topics from the given text using Latent Dirichlet Allocation (LDA).

    Args:
        text (str): The input text.
        num_topics (int): The number of topics to extract.
        num_words (int): The number of words to represent each topic.

    Returns:
        List[str]: The extracted topics represented by their top words.
    """
    try:
        # Preprocess the text
        filtered_tokens = preprocess_text(text)
        # Create a dictionary and corpus
        dictionary = corpora.Dictionary([filtered_tokens])
        corpus = [dictionary.doc2bow(filtered_tokens)]
        # Train the LDA model
        lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=4)
        # Extract the topics
        topics = []
        for topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
            topic_words = topic[1].split('+')
            topic_words = [word.split('*')[1].strip('"') for word in topic_words]
            topics.extend(topic_words)
        return topics
    except Exception as e:
        logging.error(f"Error in extract_topics: {str(e)}")
        return []

def get_topic_clusters(conversation_history: List[Dict[str, str]], num_topics: int = 5) -> List[List[Dict[str, str]]]:
    """
    Get topic clusters from the conversation history using Latent Dirichlet Allocation (LDA).

    Args:
        conversation_history (List[Dict[str, str]]): The history of the conversation.
        num_topics (int): The number of topics to extract.

    Returns:
        List[List[Dict[str, str]]]: The topic clusters, where each cluster is a list of conversation turns.
    """
    if not conversation_history:
        return []  # Return an empty list if conversation_history is empty
    
    texts = [conv['response'] for conv in conversation_history]
    preprocessed_texts = [preprocess_text(text) for text in texts]
    dictionary = corpora.Dictionary(preprocessed_texts)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]
    
    if not dictionary.token2id:
        return []  # Return an empty list if there are no valid terms in the dictionary
    
    num_topics = min(num_topics, len(dictionary))  # Adjust the number of topics based on the size of the dictionary
    
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, workers=4)
    
    topic_clusters = [[] for _ in range(num_topics)]
    for i, conv in enumerate(conversation_history):
        bow_vector = dictionary.doc2bow(preprocess_text(conv['response']))
        topic_probabilities = lda_model.get_document_topics(bow_vector)
        top_topic_index = max(topic_probabilities, key=lambda x: x[1])[0]
        topic_clusters[top_topic_index].append(conv)
    
    # Evaluate the quality of extracted topics
    topic_coherence_scores = []
    for topic_words in lda_model.print_topics(num_topics=num_topics, num_words=10):
        topic_words = [word.split('*')[1].strip('"') for word in topic_words[1].split('+')]
        coherence_score = evaluate_topic_coherence(topic_words)
        topic_coherence_scores.append(coherence_score)
    
    avg_topic_coherence = sum(topic_coherence_scores) / len(topic_coherence_scores)
    logging.info(f"Average Topic Coherence Score: {avg_topic_coherence:.2f}")

    return topic_clusters