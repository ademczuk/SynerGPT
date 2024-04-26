# test_API.py
import os
import openai
from anthropic import Client, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

load_dotenv(dotenv_path='_API_Keys.env')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Create Anthropic client
client = Client(api_key=ANTHROPIC_API_KEY)

# Set prompt
prompt = "Hello, how are you?"

try:
    # Send prompt to Claude
    response = client.completions.create(
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
        max_tokens_to_sample=550,
        model="claude-v1.3"
    )
    # Print the generated response
    generated_response = response.completion.strip()
    print("Claude's response:", generated_response)
except Exception as e:
    print(f"An error occurred: {e}")

# Pin the openai library to version 0.27.0
openai.api_key = os.getenv('OPENAI_API_KEY')
prompt = "Hello, how are you?"

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    generated_text = response.choices[0].message.content.strip()
    print(f"OpenAI response: {generated_text}")
except Exception as e:
    print(f"An error occurred: {e}")