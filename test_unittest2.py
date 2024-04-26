import unittest
from unittest.mock import patch, MagicMock
import json
from control import app, interact, main
from pymongo import MongoClient
from data_handling import save_dataset, load_dataset
from train_model import train_model, fine_tune_model, load_and_preprocess_data
from transformers import BertTokenizer
from ClaudeManager import ClaudeManager
from ChatGPTWorker import ChatGPTWorker
from context_aware_modeling import ContextAwareModel
from reinforcement_learning import DynamicPlanner
from feedback_utils import get_llm_feedback, save_to_labeled_dataset
from semantic_similarity import calculate_semantic_similarity
from topic_modeling import extract_topics
from keyword_extraction import extract_keywords
from evaluation import evaluate_conversation, evaluate_sentiment_classification
from dataset import LabeledDataset
import numpy as np

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.mongo_client = MongoClient()
        self.db = self.mongo_client['conversation_db']
        self.collection = self.db['dataset']
        self.context_model = ContextAwareModel()
        self.dynamic_planner = DynamicPlanner(state_dim=100, action_dim=4, device='cpu')

    def tearDown(self):
        self.collection.delete_many({})
        self.mongo_client.close()

    def test_interact_success(self):
        print("Testing successful interaction...")
        # Test case to verify a successful interaction with the API
        data = {
            'prompt': 'Test prompt',
            'max_cycles': 3
        }
        response = self.app.post('/api/interact', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['message'], 'Interaction completed successfully.')
        print("Successful interaction test passed.")

    def test_interact_missing_prompt(self):
        print("Testing interaction with missing prompt...")
        # Test case to verify the API's response when the prompt is missing
        data = {
            'max_cycles': 3
        }
        response = self.app.post('/api/interact', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error'], 'Prompt is required.')
        print("Missing prompt test passed.")

    def test_interact_invalid_max_cycles(self):
        print("Testing interaction with invalid max_cycles...")
        # Test case to verify the API's response when max_cycles is invalid
        data = {
            'prompt': 'Test prompt',
            'max_cycles': 'invalid'
        }
        response = self.app.post('/api/interact', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error'], 'Max cycles must be a valid integer.')
        print("Invalid max_cycles test passed.")

    def test_save_and_load_dataset(self):
        print("Testing saving and loading dataset...")
        # Test case to verify the saving and loading of the dataset using MongoDB
        dataset = [
            {'text': 'Sample text 1', 'feedback': 'Sample feedback 1'},
            {'text': 'Sample text 2', 'feedback': 'Sample feedback 2'}
        ]
        save_dataset(dataset)
        loaded_dataset = load_dataset()
        self.assertEqual(loaded_dataset, dataset)
        print("Save and load dataset test passed.")

    def test_train_model(self):
        print("Testing model training...")
        # Test case to verify the training of the model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_texts = ['Sample text 1', 'Sample text 2']
        train_sentiments = [0, 1]
        train_relevance = [0.8, 0.6]
        train_insightfulness = [0.7, 0.9]
        val_texts = ['Sample text 3']
        val_sentiments = [1]
        val_relevance = [0.5]
        val_insightfulness = [0.8]
        
        train_dataset = LabeledDataset(train_texts, train_sentiments, train_relevance, train_insightfulness, tokenizer)
        val_dataset = LabeledDataset(val_texts, val_sentiments, val_relevance, val_insightfulness, tokenizer)
        
        with patch('control.logging') as mock_logging:
            model = train_model(train_dataset, val_dataset, tokenizer)
            mock_logging.info.assert_called_with('Model training completed.')
        print("Model training test passed.")

    def test_claude_generate_prompt(self):
        print("Testing Claude prompt generation...")
        # Test case to verify the generation of a prompt by Claude
        claude_manager = ClaudeManager(self.dynamic_planner)
        previous_response = "Previous response"
        conversation_history = [{"prompt": "Test prompt", "response": "Test response"}]
        original_prompt = "Original prompt"
        
        with patch('ClaudeManager.client.completions.create') as mock_create:
            mock_response = MagicMock()
            mock_response.completion.strip.return_value = "Generated prompt"
            mock_create.return_value = mock_response
            
            generated_prompt = claude_manager.generate_prompt(previous_response, conversation_history, self.context_model, original_prompt)
            self.assertEqual(generated_prompt, "Generated prompt")
        print("Claude prompt generation test passed.")

    def test_chatgpt_generate_response(self):
        print("Testing ChatGPT response generation...")
        # Test case to verify the generation of a response by ChatGPT
        chatgpt_worker = ChatGPTWorker()
        prompt = "Test prompt"
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_response = MagicMock()
            mock_response.choices[0].message.content.strip.return_value = "Generated response"
            mock_create.return_value = mock_response
            
            generated_response = chatgpt_worker.generate_response(prompt, "Assistant", self.context_model, self.dynamic_planner)
            self.assertEqual(generated_response, "Generated response")
        print("ChatGPT response generation test passed.")

    def test_interaction_between_apis(self):
        print("Testing interaction between Claude and ChatGPT...")
        # Test case to verify the interaction between Claude and ChatGPT
        claude_manager = ClaudeManager(self.dynamic_planner)
        chatgpt_worker = ChatGPTWorker()
        prompt = "Test prompt"
        max_cycles = 2
        
        with patch('ClaudeManager.client.completions.create') as mock_claude_create, \
             patch('openai.ChatCompletion.create') as mock_chatgpt_create:
            mock_claude_response = MagicMock()
            mock_claude_response.completion.strip.return_value = "Claude response"
            mock_claude_create.return_value = mock_claude_response
            
            mock_chatgpt_response = MagicMock()
            mock_chatgpt_response.choices[0].message.content.strip.return_value = "ChatGPT response"
            mock_chatgpt_create.return_value = mock_chatgpt_response
            
            main(prompt, max_cycles)
            self.assertEqual(mock_claude_create.call_count, max_cycles // 2)
            self.assertEqual(mock_chatgpt_create.call_count, max_cycles // 2)
        print("Interaction between Claude and ChatGPT test passed.")

    def test_get_llm_feedback(self):
        print("Testing LLM feedback generation...")
        # Test case to verify the generation of LLM feedback
        response = "Test response"
        llm_name = "Claude"
        previous_scores = (0.5, 0.8)
        
        with patch('feedback_utils.BertForSequenceClassification.from_pretrained') as mock_model, \
             patch('feedback_utils.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            
            sentiment, relevance, insightfulness = get_llm_feedback(response, llm_name, previous_scores)
            self.assertIsInstance(sentiment, str)
            self.assertIsInstance(relevance, float)
            self.assertIsInstance(insightfulness, float)
        print("LLM feedback generation test passed.")

    def test_save_to_labeled_dataset(self):
        print("Testing saving to labeled dataset...")
        # Test case to verify saving data to the labeled dataset
        response = "Test response"
        feedback = "Test feedback"
        responder_name = "ChatGPT"
        
        with patch('feedback_utils.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            mock_open.return_value = MagicMock()
            
            save_to_labeled_dataset(response, feedback, self.context_model, responder_name, self.dynamic_planner)
            mock_open.assert_called_once_with('labeled_dataset.json', 'w')
            mock_json_dump.assert_called_once()
        print("Saving to labeled dataset test passed.")

    def test_calculate_semantic_similarity(self):
        print("Testing semantic similarity calculation...")
        # Test case to verify the calculation of semantic similarity between texts
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        
        with patch('semantic_similarity.sentence_transformer.encode') as mock_encode:
            mock_encode.return_value = [0.1, 0.2, 0.3]
            
            similarity = calculate_semantic_similarity(text1, text2)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
        print("Semantic similarity calculation test passed.")

    def test_extract_topics(self):
        print("Testing topic extraction...")
        # Test case to verify the extraction of topics from text
        text = "This is a sample text about artificial intelligence and machine learning."
        
        with patch('topic_modeling.preprocess_text') as mock_preprocess, \
             patch('topic_modeling.corpora.Dictionary') as mock_dictionary, \
             patch('topic_modeling.LdaMulticore') as mock_lda:
            mock_preprocess.return_value = ["sample", "text", "artificial", "intelligence", "machine", "learning"]
            mock_dictionary.return_value = MagicMock()
            mock_lda.return_value = MagicMock()
            mock_lda.return_value.print_topics.return_value = [("topic1", "0.1*word1 + 0.2*word2"), ("topic2", "0.3*word3 + 0.4*word4")]
            
            topics = extract_topics(text)
            self.assertIsInstance(topics, list)
            self.assertGreater(len(topics), 0)
        print("Topic extraction test passed.")

    def test_extract_keywords(self):
        print("Testing keyword extraction...")
        # Test case to verify the extraction of keywords from text
        text = "This is a sample text about artificial intelligence and machine learning."
        
        with patch('keyword_extraction.nlp') as mock_nlp:
            mock_doc = MagicMock()
            mock_doc.noun_chunks = [MagicMock(text="artificial intelligence"), MagicMock(text="machine learning")]
            mock_nlp.return_value = mock_doc
            
            keywords = extract_keywords(text)
            self.assertIsInstance(keywords, list)
            self.assertGreater(len(keywords), 0)
        print("Keyword extraction test passed.")

    def test_context_aware_modeling(self):
        print("Testing context-aware modeling...")
        # Test case to verify the functionality of context-aware modeling
        prompt = "What is artificial intelligence?"
        response = "Artificial intelligence is the simulation of human intelligence in machines."
        feedback = "Relevance: 4, Insightfulness: 3"
        
        self.context_model.update_context(prompt, response, feedback)
        context = self.context_model.get_context()
        self.assertIsInstance(context, dict)
        self.assertIn("topic", context)
        
        relevant_history = self.context_model.get_relevant_history(self.context_model.conversation_history, prompt)
        self.assertIsInstance(relevant_history, list)
        print("Context-aware modeling test passed.")

    def test_reinforcement_learning(self):
        print("Testing reinforcement learning...")
        # Test case to verify the functionality of reinforcement learning
        state = self.dynamic_planner.get_state("Prompt", "Response")
        self.assertIsInstance(state, np.ndarray)
        
        action = self.dynamic_planner.select_action(state)
        self.assertIn(action, ['expand', 'clarify', 'summarize', 'evaluate'])
        
        reward = self.dynamic_planner.get_reward("Prompt", "Response")
        self.assertIsInstance(reward, float)
        
        next_state = self.dynamic_planner.get_state("Prompt", "Response")
        self.assertIsInstance(next_state, np.ndarray)
        
        done = 1 if reward >= 4.5 else 0
        self.assertIn(done, [0, 1])
        
        self.dynamic_planner.update_model(state, action, reward, next_state, done)
        print("Reinforcement learning test passed.")

    def test_fallback_model(self):
        print("Testing fallback model...")
        # Test case to verify the functionality of the fallback model
        claude_manager = ClaudeManager(self.dynamic_planner)
        claude_manager.is_model_loaded = False
        
        previous_response = "Previous response"
        conversation_history = [{"prompt": "Test prompt", "response": "Test response"}]
        original_prompt = "Original prompt"
        
        with patch('fallback_models.FallbackModel.generate_prompt') as mock_generate_prompt:
            mock_generate_prompt.return_value = "Fallback prompt"
            
            generated_prompt = claude_manager.generate_prompt(previous_response, conversation_history, self.context_model, original_prompt)
            self.assertEqual(generated_prompt, "Fallback prompt")
        print("Fallback model test passed.")

    def test_evaluate_conversation(self):
        print("Testing conversation evaluation...")
        # Test case to verify the evaluation of a conversation
        conversation_history = [
            {"prompt": "What is the meaning of life?", "response": "The meaning of life is subjective and varies for each individual."},
            {"prompt": "How can I find happiness?", "response": "Happiness can be found through various means such as practicing gratitude, cultivating relationships, and pursuing meaningful goals."}
        ]
        
        with patch('evaluation.get_topic_clusters') as mock_get_topic_clusters, \
             patch('evaluation.calculate_semantic_similarity') as mock_calculate_similarity:
            mock_get_topic_clusters.return_value = [conversation_history]
            mock_calculate_similarity.return_value = 0.8
            
            relevance_score, coherence_score = evaluate_conversation(conversation_history)
            self.assertIsInstance(relevance_score, float)
            self.assertIsInstance(coherence_score, float)
        print("Conversation evaluation test passed.")

    def test_evaluate_sentiment_classification(self):
        print("Testing sentiment classification evaluation...")
        # Test case to verify the evaluation of sentiment classification
        true_sentiments = [0, 1, 2, 1, 0]
        predicted_sentiments = [0, 1, 1, 1, 0]
        
        accuracy, precision, recall, f1 = evaluate_sentiment_classification(true_sentiments, predicted_sentiments)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(f1, float)
        print("Sentiment classification evaluation test passed.")

    def test_model_finetuning(self):
        print("Testing model fine-tuning...")
        # Test case to verify the fine-tuning of the model
        
        fine_tune_script_path = 'train_model.py'
        model_finetuned_path = './model_finetuned/'
        labeled_data = [
            {'text': 'Sample text 1', 'feedback': 'Sentiment: Positive, Relevance: 4, Insightfulness: 3'},
            {'text': 'Sample text 2', 'feedback': 'Sentiment: Negative, Relevance: 2, Insightfulness: 4'}
        ]
        
        with patch('train_model.load_dataset') as mock_load_dataset, \
                patch('train_model.train_model') as mock_train_model:
            mock_load_dataset.return_value = labeled_data
            mock_train_model.return_value = MagicMock()
            
            fine_tune_model(fine_tune_script_path, model_finetuned_path)
            mock_load_dataset.assert_called_once()
            mock_train_model.assert_called_once()
        print("Model fine-tuning test passed.")

    def test_data_preprocessing(self):
        print("Testing data preprocessing...")
        # Test case to verify the preprocessing of data
        texts = ["This is a sample text.", "Another example text."]
        sentiments = ["Positive", "Negative"]
        relevance_scores = [0.8, 0.6]
        insightfulness_scores = [0.7, 0.9]
        
        train_texts, val_texts, train_sentiments, val_sentiments, train_relevance, val_relevance, train_insightfulness, val_insightfulness = load_and_preprocess_data(texts, sentiments, relevance_scores, insightfulness_scores)
        
        self.assertIsInstance(train_texts, list)
        self.assertIsInstance(val_texts, list)
        self.assertIsInstance(train_sentiments, list)
        self.assertIsInstance(val_sentiments, list)
        self.assertIsInstance(train_relevance, list)
        self.assertIsInstance(val_relevance, list)
        self.assertIsInstance(train_insightfulness, list)
        self.assertIsInstance(val_insightfulness, list)
        print("Data preprocessing test passed.")

    def test_api_error_handling(self):
        print("Testing API error handling...")
        # Test case to verify error handling in the API
        data = {
            'prompt': 'Test prompt',
            'max_cycles': 'invalid'
        }
        
        with patch('control.main') as mock_main:
            mock_main.side_effect = Exception("Mocked API error")
            
            response = self.app.post('/api/interact', data=json.dumps(data), content_type='application/json')
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json['error'], 'An unexpected error occurred.')
        print("API error handling test passed.")

# Add more test cases as needed

if __name__ == '__main__':
   unittest.main()