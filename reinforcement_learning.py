import random
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DynamicPlanner:
    def __init__(self):
        self.current_plan = []
        self.reward_history = []

    def adapt_prompt(self, prompt, previous_response, conversation_history):
        # Implement your dynamic planning logic here
        # Adapt the prompt based on the previous response, conversation history, and current plan
        adapted_prompt = prompt

        # Example: Introduce new topics or angles based on the conversation flow
        if len(conversation_history) > 2 and random.random() < 0.3:
            new_topic = self.generate_new_topic(conversation_history)
            adapted_prompt += f"\nLet's also explore a new topic: {new_topic}"

        self.current_plan.append(adapted_prompt)
        return adapted_prompt

    def update_plan(self, prompt, response, feedback, relevance_score, insightfulness_score):
        # Update the current plan based on the response, feedback, and scores
        self.current_plan.append(response)
        reward = relevance_score + insightfulness_score
        self.reward_history.append(reward)

        # Perform any necessary adjustments to the plan based on the reward
        if len(self.reward_history) > 3:
            avg_reward = sum(self.reward_history[-3:]) / 3
            if avg_reward < 3.0:
                # Adjust the plan if the average reward is low
                self.current_plan = self.current_plan[:-1]  # Remove the last response from the plan
                logging.info("Adjusted the plan due to low average reward.")

    def get_current_plan(self):
        return self.current_plan

    def generate_new_topic(self, conversation_history):
        # Implement your logic to generate a new topic based on the conversation history
        # This can be based on semantic similarity, keyword extraction, or other techniques
        new_topic = "New topic generated based on conversation history"
        return new_topic