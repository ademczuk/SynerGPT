# test_imports.py
"""
Test module for importing various modules from the SynerGPT project.
"""
import importlib
import os
import sys

# Local imports
from context_aware_modeling import ContextAwareModel
from semantic_similarity import calculate_semantic_similarity
from sentiment_analysis import analyze_sentiment
from sentiment_module import SentimentAnalyzer

# Get the current working directory
cwd = os.getcwd()

# Append the parent directory to the Python path
sys.path.append(os.path.dirname(cwd))

try:
    from utils import extract_keywords
    print("Import 'SynerGPT.utils.utils' successful")
except ImportError as e:
    print(f"Error importing 'SynerGPT.utils.utils': {str(e)}")

# Attempt to import the module using importlib
try:
    syner_gpt_spec = importlib.util.find_spec("SynerGPT")
    if syner_gpt_spec is None:
        print("Module 'SynerGPT' not found")
    else:
        syner_gpt = importlib.util.module_from_spec(syner_gpt_spec)
        syner_gpt_spec.loader.exec_module(syner_gpt)
        print("Module 'SynerGPT' imported successfully")
except ImportError as e:
    print(f"Error importing 'SynerGPT' using importlib: {str(e)}")

# Use the imported modules/functions to avoid unused import warnings
ContextAwareModel()
calculate_semantic_similarity(
    "sample text", "another sample text"
)
analyze_sentiment("This is a positive sentence.")
sentiment_analyzer = SentimentAnalyzer()
