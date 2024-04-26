# question_answering.py
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def answer_question(question: str, context: str) -> str:
    try:
        # Load question answering model
        qa_model = pipeline("question-answering")
        # Generate answer
        result = qa_model(question=question, context=context)
        answer = result['answer']
        return answer
    except Exception as e:
        logging.error(f"Error in answer_question: {str(e)}")
        return "Unable to generate an answer."