# refine_prompt.py
from reinforcement_learning import DynamicPlanner, evaluate_insights
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import random
from context_aware_modeling import ContextAwareModel
from nltk.corpus import wordnet
import nltk
from semantic_similarity import calculate_semantic_similarity
from utils import extract_keywords
from sentiment_analysis import analyze_sentiment
from sentiment_module import SentimentAnalyzer
from logging_config import LOG_LEVEL, LOG_FORMAT


nltk.data.path.append(r'C:\Users\Joey\AppData\Roaming\nltk_data')

# Download the required NLTK resources
#if not nltk.data.find('corpora/wordnet'):
#    nltk.download('wordnet')

# Configure logging
logging.basicConfig(filename='app.log', level=LOG_LEVEL, format=LOG_FORMAT)
#init_logger()
logger = logging.getLogger(__name__)

def get_sentiment(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device) -> str:
    """
    Get the sentiment of the given text using the provided model and tokenizer.

    Args:
        text (str): The text to analyze.
        model (AutoModelForSequenceClassification): The sentiment analysis model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        device (torch.device): The device to run the model on.

    Returns:
        str: The predicted sentiment (negative, neutral, or positive).
    """
    try:
        if not model or not tokenizer:
            logging.error("Model or tokenizer not loaded correctly.")
            return "neutral"
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            if not outputs:
                logging.error("Error in model outputs.")
                return "neutral"
            predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()
        
        sentiment_labels = ["negative", "neutral", "positive"]
        if predicted_sentiment >= len(sentiment_labels):
            logging.error(f"Invalid sentiment index: {predicted_sentiment}")
            return "neutral"
        
        return sentiment_labels[predicted_sentiment]
    except Exception as e:
        logging.error(f"Error in get_sentiment: {str(e)}")
        return "neutral"

def generate_example_scenario(topic: str) -> str:
    """
    Generate an example scenario based on the given topic.

    Args:
        topic (str): The topic for the example scenario.

    Returns:
        str: The generated example scenario.
    """
    examples = {
        "artificial intelligence": [
            "Imagine a world where AI assists humans in every aspect of life. Discuss the potential benefits and risks.",
            "Consider the ethical implications of AI making decisions that impact human lives. How can we ensure fairness and transparency?",
            "Explore the role of AI in healthcare. How can it revolutionize patient diagnosis and treatment?",
            "Discuss the impact of AI on the workforce. What jobs are most likely to be automated, and how can society adapt?",
            "Imagine AI being used in the education sector. How can it personalize learning and improve student outcomes?"
        ],
        "climate change": [
            "Suppose you are a policymaker tasked with developing a plan to combat climate change. What strategies would you prioritize?",
            "Imagine a future where renewable energy has replaced fossil fuels. Describe the challenges and opportunities in this transition.",
            "Discuss the role of individual actions in mitigating climate change. What can people do in their daily lives to make a difference?",
            "Explore the potential impact of climate change on global food security. How can we ensure a sustainable food supply?",
            "Consider the relationship between climate change and social inequality. How can we address the disproportionate impacts on vulnerable communities?"
        ],
        "mental health": [
            "Discuss the stigma surrounding mental health issues. How can society work to break down these barriers?",
            "Explore the impact of social media on mental health. What are the potential benefits and drawbacks?",
            "Imagine a world where mental health services are easily accessible to everyone. What would that look like?",
            "Consider the relationship between mental health and the workplace. How can employers support their employees' well-being?",
            "Discuss the role of technology in mental health treatment. What innovations show promise in improving care?"
        ],
        "space exploration": [
            "Imagine humanity has established a colony on Mars. What challenges would they face, and how could they overcome them?",
            "Discuss the potential benefits of space exploration for scientific research and technological advancement.",
            "Consider the ethical implications of sending humans on long-duration space missions. How can we ensure their well-being?",
            "Explore the possibility of discovering extraterrestrial life. What would be the implications for humanity?",
            "Imagine space tourism becomes widely accessible. What impact would this have on society and the environment?"
        ],
        "cybersecurity": [
            "Discuss the growing threat of cyberattacks on individuals, businesses, and governments. How can we improve our defenses?",
            "Explore the ethical considerations surrounding data privacy and security. How can we balance privacy with the need for information sharing?",
            "Imagine a future where quantum computing renders current encryption methods obsolete. What new security measures would be needed?",
            "Consider the role of human error in cybersecurity breaches. How can we create a culture of cybersecurity awareness?",
            "Discuss the potential impact of artificial intelligence on cybersecurity. How can AI be used to enhance or undermine security measures?"
        ]
    }
    return random.choice(examples.get(topic, [""]))

def get_synonyms(word: str) -> List[str]:
    """
    Get synonyms for the given word using WordNet.

    Args:
        word (str): The word to get synonyms for.

    Returns:
        List[str]: A list of synonyms for the given word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def refine_prompt(previous_response: str, llm_name: str, conversation_history: List[dict], context_model: ContextAwareModel, original_prompt: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device, dynamic_planner: DynamicPlanner) -> str:
    try:
        # Adapt the prompt using the dynamic_planner
        adapted_prompt = dynamic_planner.adapt_prompt(previous_response, conversation_history)
        logging.info(f"Adapted prompt: {adapted_prompt}")

        topic_keywords = ', '.join(extract_keywords(previous_response))
        logging.info(f"Topic keywords: {topic_keywords}")

        tone = get_sentiment(previous_response, model, tokenizer, device)
        logging.info(f"Tone: {tone}")

        example_scenario = generate_example_scenario(topic_keywords)
        logging.info(f"Example scenario: {example_scenario}")

        if llm_name == "Claude":
            refined_prompt = f"Considering the {tone} tone and focusing on topics: {topic_keywords}, guide ChatGPT to build upon the adapted prompt: {adapted_prompt}. Encourage exploring multiple perspectives and generating unique ideas to address the question. Here's an example scenario to consider:\n{example_scenario}"
        else:
            refined_prompt = f"Considering the {tone} tone and focusing on topics: {topic_keywords}, generate a thoughtful and comprehensive response that directly addresses the adapted prompt: {adapted_prompt}. Provide specific information and examples to answer the question. Here's an example scenario to consider:\n{example_scenario}"

        # Incorporate synonym replacement for keyword expansion
        expanded_keywords = []
        for keyword in topic_keywords.split(', '):
            synonyms = get_synonyms(keyword)
            if synonyms:
                expanded_keywords.append(random.choice(synonyms))
            else:
                expanded_keywords.append(keyword)
        expanded_topic_keywords = ', '.join(expanded_keywords)

        # Evaluate the effectiveness of synonym replacement
        original_similarity = calculate_semantic_similarity(topic_keywords, original_prompt)
        expanded_similarity = calculate_semantic_similarity(expanded_topic_keywords, original_prompt)
        if expanded_similarity < original_similarity:
            # Revert to the original topic keywords if the expanded keywords reduce similarity
            expanded_topic_keywords = topic_keywords

        refined_prompt = refined_prompt.replace(topic_keywords, expanded_topic_keywords)

        # Update the dynamic_planner with state, action, reward, and next_state
        state = dynamic_planner.get_state(previous_response, refined_prompt)
        action = dynamic_planner.select_action(state)
        reward = dynamic_planner.get_reward(previous_response, refined_prompt)
        next_state = dynamic_planner.get_state(previous_response, refined_prompt)
        done = 1 if reward >= 4.5 else 0
        dynamic_planner.update_model(state, action, reward, next_state, done)

        # Update the dynamic_planner's plan based on relevance and insightfulness scores
        relevance_score = calculate_semantic_similarity(refined_prompt, original_prompt)
        insightfulness_score = evaluate_insights(refined_prompt)
        dynamic_planner.update_plan(previous_response, refined_prompt, None, relevance_score, insightfulness_score)

        logging.info(f"Refined prompt: {refined_prompt}")

        return refined_prompt
    except Exception as e:
        logging.error(f"Error in refine_prompt: {str(e)}")
        return previous_response