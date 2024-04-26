# test_importlib.py

import importlib.util

modules_to_check = [
    'utils',
    'keyword_extraction',
    'semantic_similarity',
    'sentiment_analysis',
    'sentiment_module',
    'topic_modeling',
    'keyword_extraction',
    'evaluation',
    'feedback_utils',
    'data_handling',
    'context_aware_modeling',
    'reinforcement_learning',
    'fallback_models',
    'ClaudeManager',
    'ChatGPTWorker',
    'analyze_response',
    'ANtalk',
    'Control',
    'dataset',
    'main',
    'question_answering',
    'Optalk',
    'train_model',
]

for module in modules_to_check:
    try:
        spec = importlib.util.find_spec(module)
        if spec is None:
            print(f"Module '{module}' not found")
        else:
            print(f"Module '{module}' found")
    except Exception as e:
        print(f"Error importing module '{module}': {str(e)}")