# refine_prompt.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from keyword_extraction import extract_keywords
import logging
import random

from context_aware_modeling import ContextAwareModel

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sentiment(text):
    try:
        model_path = './model_finetuned/'
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_sentiment = torch.argmax(outputs.logits, dim=1).item()

        sentiment_labels = ["negative", "neutral", "positive"]
        return sentiment_labels[predicted_sentiment]
    except Exception as e:
        logging.error(f"Error in get_sentiment: {str(e)}")
        return "neutral"

def generate_example_scenario(topic):
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

def refine_prompt(previous_response, llm_name, conversation_history, context_model):
    try:
        topic_keywords = ', '.join(extract_keywords(previous_response))
        tone = get_sentiment(previous_response)
        example_scenario = generate_example_scenario(topic_keywords)

        if llm_name == "Claude":
            refined_prompt = f"Considering the {tone} tone and focusing on topics: {topic_keywords}, guide ChatGPT to build upon the previous response and provide more detailed insights related to improving the prompt. Encourage exploring multiple perspectives and generating unique ideas to enhance the prompt. Here's an example scenario to consider:\n{example_scenario}"
        else:
            refined_prompt = f"Considering the {tone} tone and focusing on topics: {topic_keywords}, generate a thoughtful and comprehensive response that expands upon the previous discussion and offers insights on how to improve the prompt. Provide specific suggestions and examples to refine the prompt. Here's an example scenario to consider:\n{example_scenario}"

        # Incorporate relevant information from conversation history
        relevant_history = context_model.get_relevant_history(conversation_history, topic_keywords)
        if relevant_history:
            refined_prompt += f"\nAdditional context from previous conversation:\n{relevant_history}"

        return refined_prompt
    except Exception as e:
        logging.error(f"Error in refine_prompt: {str(e)}")
        return previous_response