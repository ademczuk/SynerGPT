import subprocess
import re
import sys
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INITIATION_PROMPT = """
*** START: Here is the user's PROMPT: 
"""

def run_script(script_name: str, prompt: str) -> str:
    try:
        process = subprocess.Popen(
            [sys.executable, script_name, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate()
        if error:
            logging.error(f"Error running {script_name}: {error}")
        return output.strip()
    except Exception as e:
        logging.error(f"Error running {script_name}: {e}")
        return ""

def extract_prompt(chat_log: str) -> str:
    prompt_pattern = r'Response:\s*(.+)'
    prompts = re.findall(prompt_pattern, chat_log, re.DOTALL)
    if prompts:
        lines = chat_log.strip().split('\n')
        for line in reversed(lines):
            if line.startswith('Response:'):
                return line.replace('Response:', '').strip()
    return ""

def get_llm_name(cycle_count: int) -> str:
    return "Claude" if cycle_count % 2 == 0 else "ChatGPT"

def get_initial_prompt_and_max_cycles() -> Tuple[str, int]:
    if len(sys.argv) == 3:
        initial_prompt = sys.argv[1]
        try:
            max_cycles = int(sys.argv[2])
        except ValueError:
            logging.warning("Invalid max_cycles argument. Using default value.")
            max_cycles = 0
    else:
        initial_prompt = input("Please enter your prompt: ")
        while True:
            try:
                max_cycles = int(input("How many cycles to allow? "))
                break
            except ValueError:
                logging.warning("Invalid input. Please enter a valid integer.")
    return initial_prompt, max_cycles

def main():
    initial_prompt, max_cycles = get_initial_prompt_and_max_cycles()
    cycle_count = 0

    while cycle_count < max_cycles:
        prompt = ""

        if cycle_count == 0:
            prompt = (
                f"\nWe will have: {max_cycles} Prompt/Response cycles to complete this task.\n"
                f"{INITIATION_PROMPT}\n{initial_prompt}\n"
            )
            output = run_script('ANtalk.py', prompt)
        else:
            with open('ChatLog.txt', 'r') as file:
                chat_log = file.read()
            prompt = extract_prompt(chat_log)
            script_name = 'ANtalk.py' if cycle_count % 2 == 0 else 'OPtalk.py'
            output = run_script(script_name, prompt)

        llm_name = get_llm_name(cycle_count)

        try:
            with open('ChatLog.txt', 'a') as file:
                file.write(f"\nCycle #: {cycle_count}\n{llm_name}\nPrompt: {prompt}\nResponse: {output}\n")
        except Exception as e:
            logging.error(f"Error writing to ChatLog.txt: {e}")

        logging.info(f"\nCycle #: {cycle_count}\n{llm_name}\nPrompt: {prompt}\nResponse: {output}\n")

        cycle_count += 1

if __name__ == '__main__':
    main()