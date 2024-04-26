# prompt_generation.py
from typing import List, Dict
from sentiment_analysis import analyze_sentiment
from context_aware_modeling import ContextAwareModel
from refine_prompt import refine_prompt

def generate_prompt(
    previous_response: str,
    conversation_history: List[Dict[str, str]],
    context_model: ContextAwareModel,
    original_prompt: str,
    model,
    tokenizer,
    device
) -> str:
    try:
        sentiment = analyze_sentiment(previous_response)
        relevant_history = context_model.get_relevant_history(conversation_history, original_prompt)
        refined_prompt = refine_prompt(previous_response, "Claude", conversation_history, context_model, original_prompt, model, tokenizer, device)

        if sentiment == "positive":
            prompt_template = "Great! Let's continue our discussion on a positive note. Considering the relevant context:\n{relevant_history}\n{refined_prompt}"
        elif sentiment == "negative":
            prompt_template = "I understand your concerns. Let's try to approach this from a different perspective. Considering the relevant context:\n{relevant_history}\n{refined_prompt}"
        else:
            prompt_template = "Thank you for sharing your thoughts. Let's delve deeper into the topic. Considering the relevant context:\n{relevant_history}\n{refined_prompt}"

        new_prompt = prompt_template.format(relevant_history=relevant_history, refined_prompt=refined_prompt)
        return new_prompt
    except Exception as e:
        print(f"Error generating prompt: {str(e)}")
        return "Error: Unable to generate prompt."