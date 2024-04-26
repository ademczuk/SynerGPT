# reinforcement_learning.py
from sentiment_analysis import analyze_sentiment
import random
import logging
from semantic_similarity import calculate_semantic_similarity
from sentiment_module import SentimentAnalyzer
from utils import extract_keywords
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, NamedTuple
from topic_modeling import extract_topics
from topic_modeling import lda_topic_modeling
from context_aware_modeling import ContextAwareModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
from semantic_similarity import calculate_semantic_similarity

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the Deep Q-Network (DQN).

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DQN.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output Q-values.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Initialize the Replay Buffer.

        Args:
            capacity (int): The maximum capacity of the replay buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new experience to the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the sampled states, actions, rewards, next states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self) -> int:
        """
        Get the current size of the replay buffer.

        Returns:
            int: The current size of the replay buffer.
        """
        return len(self.buffer)

class DynamicPlanner:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        """
        Initialize the Dynamic Planner.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            device (torch.device): The device to run the model on.
        """
        self.device = device
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update_frequency = 1000
        self.update_counter = 0
        self.current_plan = []
        self.conversation_history = []  # Initialize conversation_history as an empty list
        self.context_model = ContextAwareModel()  # Initialize the context model

    def get_state(self, prompt: str, response: str) -> np.ndarray:
        """
        Get the state representation based on the prompt and response.
        Args:
            prompt (str): The prompt text.
            response (str): The response text.
        Returns:
            np.ndarray: The state representation.
        """
        state = []

        # Sentiment score
        analyzer = SentimentAnalyzer()
        sentiment = analyzer.predict_sentiment(response)
        sentiment_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        sentiment_score = sentiment_map.get(sentiment, 0.0)
        state.append(sentiment_score)

        # Coherence score
        coherence_score = self.evaluate_coherence(response, self.conversation_history)
        state.append(coherence_score)

        # Task completion score
        task_keywords = extract_keywords(prompt)
        task_completion_score = self.evaluate_task_completion(response, task_keywords)
        state.append(task_completion_score)

        # Topic keywords
        topic_keywords = extract_keywords(response)[:5]  # Limit to top 5 topic keywords
        state.extend(topic_keywords)

        # Pad or truncate the state to a fixed size
        max_state_size = 100
        state = state[:max_state_size] + [0] * (max_state_size - len(state))

        return np.array(state)

    def evaluate_coherence(self, response: str, conversation_history: List[Dict[str, str]]) -> float:
        """
        Evaluate the coherence of the response with respect to the conversation history.

        Args:
            response (str): The response text.
            conversation_history (List[Dict[str, str]]): The history of the conversation.

        Returns:
            float: The coherence score.
        """
        if not conversation_history:
            return 0.0

        similarity_scores = []
        for conv in conversation_history[-3:]:  # Consider the last 3 conversation turns
            if isinstance(conv, dict) and 'response' in conv:
                similarity_score = calculate_semantic_similarity(response, conv['response'])
            else:
                similarity_score = calculate_semantic_similarity(response, conv)
            similarity_scores.append(similarity_score)

        coherence_score = sum(similarity_scores) / len(similarity_scores)
        return coherence_score



    def evaluate_task_completion(self, response: str, task_keywords: List[str]) -> float:
        """
        Evaluate the task completion score based on the response and task keywords.

        Args:
            response (str): The response text.
            task_keywords (List[str]): The keywords related to the task.

        Returns:
            float: The task completion score.
        """
        response_keywords = extract_keywords(response)
        matching_keywords = set(response_keywords) & set(task_keywords)
        if len(task_keywords) > 0:
            task_completion_score = len(matching_keywords) / len(task_keywords)
        else:
            task_completion_score = 0.0
        return task_completion_score

    def get_reward(self, prompt: str, response: str) -> float:
        """
        Calculate the reward based on the prompt and response.
        Args:
            prompt (str): The prompt text.
            response (str): The response text.
        Returns:
            float: The calculated reward.
        """
        relevance_score = calculate_semantic_similarity(prompt, response)
        coherence_score = self.evaluate_coherence(response, self.conversation_history)
        task_completion_score = self.evaluate_task_completion(response, extract_keywords(prompt))

        # Use the SentimentAnalyzer to get the sentiment score
        analyzer = SentimentAnalyzer()
        sentiment_score = analyzer.predict_sentiment(response)
        sentiment_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        sentiment_score = sentiment_map.get(sentiment_score, 0.0)

        # Define weights for each component
        relevance_weight = 0.4
        coherence_weight = 0.3
        task_completion_weight = 0.2
        sentiment_weight = 0.1

        # Calculate the weighted reward
        reward = (
            relevance_weight * relevance_score +
            coherence_weight * coherence_score +
            task_completion_weight * task_completion_score +
            sentiment_weight * sentiment_score
        )

        return reward

    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of the given text.
        Args:
            text (str): The input text.
        Returns:
            str: The sentiment label (positive, negative, or neutral).
        """
        try:
            sentiment_score = analyze_sentiment(text)
            if sentiment_score > 0.3:
                return "positive"
            elif sentiment_score < -0.3:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            logging.error(f"Error in analyze_sentiment: {str(e)}")
            return "neutral"

    def extract_topics(self, text: str, num_topics: int = 3) -> List[str]:
        """
        Extract the main topics from the given text.
        Args:
            text (str): The input text.
            num_topics (int): The number of topics to extract.
        Returns:
            List[str]: The list of extracted topics.
        """
        try:
            topics = extract_topics(text, num_topics=num_topics)
            return topics
        except Exception as e:
            logging.error(f"Error in extract_topics: {str(e)}")
            return []

    def get_relevant_context(self, conversation_history: List[Dict[str, str]], topics: List[str]) -> str:
        """
        Retrieve relevant context from the conversation history based on the given topics.
        Args:
            conversation_history (List[Dict[str, str]]): The conversation history.
            topics (List[str]): The list of topics.
        Returns:
            str: The relevant context.
        """
        try:
            relevant_context = self.context_model.get_relevant_context(conversation_history, topics)
            return relevant_context
        except Exception as e:
            logging.error(f"Error in get_relevant_context: {str(e)}")
            return ""

    def select_action(self, state: np.ndarray) -> str:
        """
        Select an action based on the current state using an epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            str: The selected action.
        """
        if random.random() < self.epsilon:
            return random.choice(['expand', 'clarify', 'summarize', 'evaluate'])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action_index = q_values.argmax().item()
            return ['expand', 'clarify', 'summarize', 'evaluate'][action_index]

    def update_model(self):
        """
        Update the Q-network using the experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch)
        next_q_values = self.target_model(next_state_batch).max(1, keepdim=True)[0]
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def adapt_prompt(self, prompt: str, previous_response: str, conversation_history: List[dict]) -> str:
        """
        Adapt the prompt based on the previous response and conversation history.
        Args:
            prompt (str): The current prompt.
            previous_response (str): The previous response.
            conversation_history (List[dict]): The conversation history.
        Returns:
            str: The adapted prompt.
        """
        adapted_prompt = prompt

        # Extract relevant information from the previous response
        sentiment = self.analyze_sentiment(previous_response)
        topics = self.extract_topics(previous_response)

        # Retrieve relevant context from the conversation history
        relevant_context = self.get_relevant_context(conversation_history, topics)

        # Adapt the prompt based on the sentiment
        if sentiment == "positive":
            adapted_prompt = f"Great! Let's continue our discussion on {', '.join(topics)}. "
        elif sentiment == "negative":
            adapted_prompt = f"I understand your concerns regarding {', '.join(topics)}. Let's explore this further. "
        else:
            adapted_prompt = f"Thank you for your insights on {', '.join(topics)}. "

        # Incorporate relevant context into the prompt
        if relevant_context:
            adapted_prompt += f"Considering the previous context: {relevant_context}, "

        # Add a question or prompt to encourage further discussion
        adapted_prompt += "What are your thoughts on this? Can you provide more details or examples?"

        return adapted_prompt

    def update_plan(self, prompt: str, response: str, feedback: str, relevance_score: float, insightfulness_score: float):
        """
        Update the dynamic plan based on the prompt, response, feedback, and scores.

        Args:
            prompt (str): The prompt text.
            response (str): The response text.
            feedback (str): The feedback text.
            relevance_score (float): The relevance score.
            insightfulness_score (float): The insightfulness score.
        """
        state = self.get_state(prompt, response)
        action = ['expand', 'clarify', 'summarize', 'evaluate'].index(self.select_action(state))
        next_state = self.get_state(prompt, response)
        coherence_score = self.evaluate_coherence(response, [conv["response"] for conv in self.conversation_history])
        task_keywords = extract_keywords(prompt)
        task_completion_score = self.evaluate_task_completion(response, task_keywords)

        done = 1 if relevance_score >= 4.5 and insightfulness_score >= 4.5 else 0
        relevance_weight = 0.3
        insightfulness_weight = 0.3
        coherence_weight = 0.2
        task_completion_weight = 0.2
        reward = (relevance_weight * relevance_score +
                  insightfulness_weight * insightfulness_score +
                  coherence_weight * coherence_score +
                  task_completion_weight * task_completion_score)
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.update_model()
        self.current_plan.append(action)
        self.conversation_history.append({"prompt": prompt, "response": response})  # Update conversation_history

    def get_current_plan(self) -> List[int]:
        """
        Get the current plan.

        Returns:
            List[int]: The current plan as a list of action indices.
        """
        return self.current_plan

def evaluate_insights(response: str) -> float:
    """
    Evaluate the insightfulness score of the response.
    Args:
        response (str): The response text.
    Returns:
        float: The insightfulness score.
    """
    try:
        doc = nlp(response)
        # Extract named entities and their labels using Named Entity Recognition (NER)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        # Extract noun phrases using spaCy's noun_chunks
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        # Tokenize and remove stop words and punctuation
        tokens = word_tokenize(response.lower())
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
        # Perform sentiment analysis on the response
        sentiment_score = analyze_sentiment(response)
        # Perform topic modeling using LDA
        topics = lda_topic_modeling(response)
        # Calculate insightfulness score based on the number of named entities, noun phrases, unique words, sentiment score, and topics
        insight_score = len(named_entities) + len(noun_phrases) + len(set(filtered_tokens)) + abs(sentiment_score) + len(topics)
        # Normalize the score to a range of 0 to 1
        insight_score /= (len(filtered_tokens) + 2 + len(topics))
        return insight_score
    except Exception as e:
        logging.error(f"Error in evaluate_insights: {str(e)}")
        return 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 100  # Placeholder state dimension
    action_dim = 4
    dynamic_planner = DynamicPlanner(state_dim, action_dim, device)

    # Placeholder conversation loop
    conversation_history = []
    for i in range(10):
        prompt = f"Prompt {i}"
        response = f"Response {i}"
        feedback = f"Feedback {i}"
        relevance_score = calculate_semantic_similarity(prompt, response)  # Actual implementation
        insightfulness_score = evaluate_insights(response)  # Actual implementation
        adapted_prompt = dynamic_planner.adapt_prompt(prompt, response, conversation_history)
        dynamic_planner.update_plan(adapted_prompt, response, feedback, relevance_score, insightfulness_score)
        conversation_history.append({"prompt": adapted_prompt, "response": response})

if __name__ == "__main__":
    main()