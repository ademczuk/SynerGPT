# ControlMini.py
import subprocess
import sys
from typing import Tuple
import logging
import sqlite3

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INITIATION_PROMPT = """
*** START: Here is the user's PROMPT: """

def run_script(script_name: str, prompt: str) -> str:
    try:
        process = subprocess.Popen(
            [sys.executable, script_name, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=600)  # Set a timeout of 10 minutes (adjust as needed)
        
        if stderr:
            logging.error(f"Error running {script_name}: {stderr}")
        
        return stdout.strip()
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout expired while running {script_name}")
        process.terminate()
        return ""
    except Exception as e:
        logging.error(f"Error running {script_name}: {e}")
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

def save_to_database(cycle_count, llm_name, prompt, response):
    try:
        conn = sqlite3.connect('conversation_history.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS conversation
                     (cycle_count INTEGER, llm_name TEXT, prompt TEXT, response TEXT)''')
        
        c.execute("INSERT INTO conversation VALUES (?, ?, ?, ?)", (cycle_count, llm_name, prompt, response))
        
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Error saving to database: {str(e)}")

def main():
    initial_prompt, max_cycles = get_initial_prompt_and_max_cycles()
    
    if not initial_prompt:
        logging.error("No initial prompt provided.")
        sys.exit(1)
    
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
            try:
                conn = sqlite3.connect('conversation_history.db')
                c = conn.cursor()
                c.execute("SELECT prompt, response FROM conversation ORDER BY cycle_count DESC LIMIT 1")
                last_prompt, last_response = c.fetchone()
                conn.close()
                
                prompt = last_response
                script_name = 'ANtalk.py' if cycle_count % 2 == 0 else 'OPtalk.py'
                output = run_script(script_name, prompt)
            except sqlite3.Error as e:
                logging.error(f"Error retrieving last prompt and response from database: {str(e)}")
                output = ""
        
        llm_name = get_llm_name(cycle_count)
        save_to_database(cycle_count, llm_name, prompt, output)
        logging.info(f"\nCycle #: {cycle_count}\n{llm_name}\nPrompt: {prompt}\nResponse: {output}\n")
        
        cycle_count += 1

if __name__ == '__main__':
    main()