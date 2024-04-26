# test_Unittest.py
import unittest
from unittest.mock import patch, MagicMock
import torch
import nltk
#nltk.download('vader_lexicon')  # Download the lexicon data
import sys
import os


# Get the current working directory
cwd = os.getcwd()

# Append the parent directory of SynerGPT to the Python path
sys.path.append(os.path.dirname(os.path.dirname(cwd)))

from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from named_tuples import UpdateModelData

print(os.getcwd())
print(nltk.data.find(r'C:\Users\Joey\AppData\Roaming\nltk_data\sentiment\vader_lexicon'))
print(nltk.data.find('C:\\Users\\Joey\\AppData\\Roaming\\nltk_data\\sentiment\\vader_lexicon'))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy.sparse.sparsetools")

class TestAPIConnections(unittest.TestCase):
    def setUp(self):
        self.context_model = ContextAwareModel()  # Initialize before each test

        # Create a DynamicPlanner instance for testing
        state_dim = 100  # Adjust as needed
        action_dim = 4   # Adjust as needed
        device = torch.device("cpu")  # Or "cuda" if available
        self.dynamic_planner = DynamicPlanner(state_dim, action_dim, device)

    @patch('anthropic.Client')
    def test_claude_connection(self, mock_client):
        # Mock the response from Anthropic API
        mock_response = MagicMock()
        mock_response.completion.strip.return_value = "I understand your concerns. Let's try to approach this from a different perspective."
        mock_client.return_value.completions.create.return_value = mock_response

        # Create an instance of ClaudeManager and pass dynamic_planner
        claude_manager = ClaudeManager(self.dynamic_planner)  # Pass the dynamic_planner instance

        # Test generating a prompt
        prompt = "Hello, how are you?"
        response = claude_manager.generate_prompt(
            prompt, [], self.context_model, prompt
        )

        expected_response = "I understand your concerns. Let's try to approach this from a different perspective."
        self.assertTrue(response.startswith(expected_response))

    @patch('openai.ChatCompletion.create')
    def test_chatgpt_connection(self, mock_create):
        # Mock the response from OpenAI API
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content.strip.return_value = "Hello, I'm ChatGPT!"
        mock_create.return_value = mock_response

        # Create an instance of ChatGPTWorker and pass dynamic_planner
        chatgpt_worker = ChatGPTWorker()

        # Test generating a response
        prompt = "Hello, how are you?"
        response = chatgpt_worker.generate_response(prompt, "Assistant", self.context_model, self.dynamic_planner)

        self.assertEqual(response, "Hello, I'm ChatGPT!")

if __name__ == '__main__':
    unittest.main()