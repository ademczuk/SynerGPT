# test_my_code.py
import openai
import pytest
from ChatGPTWorker import ChatGPTWorker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_generate_response_empty_prompt():
    """ Test response generation with an empty prompt. """
    chatgpt_worker = ChatGPTWorker()
    try:
        with pytest.raises(ValueError):
            chatgpt_worker.generate_response("", "Assistant", None, None)
        print("Empty prompt test passed: No input allowed as expected.")
    except ValueError as e:
        print(f"Test failed: {str(e)}")

def test_generate_response_api_error(mocker):
    """Mock an API error and test the error handling."""
    # Mock the API call to simulate an API error
    mocker.patch('openai.ChatCompletion.create', side_effect=Exception("Mocked API error"))
    
    chatgpt_worker = ChatGPTWorker()
    
    with pytest.raises(Exception) as excinfo:
        chatgpt_worker.generate_response("Test prompt", "Assistant", None, None)
    assert "Mocked API error" in str(excinfo.value)

def generate_response(self, prompt, role, context_model, dynamic_planner):
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    try:
        # Simulated API call that might raise an exception
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise  # This ensures the exception is re-raised after logging
