"""
Theta AI Conversation Mode with Real-time Learning

This script allows interactive conversation with Theta, while simultaneously
training the model on the conversations in real-time.
"""

import os
import time
import json
import torch
import random
import logging
import argparse
import nltk
import re
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from model import ThetaConfig, ThetaModel
from nltk.corpus import words, wordnet
from tqdm import tqdm
from config import MODEL_DIR, MODEL_FILE
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationMemory:
    """Stores conversation history and handles conversation data processing."""
    
    def __init__(self, max_history=10, memory_file="data/theta_conversation_memory.jsonl"):
        self.history = []
        self.max_history = max_history
        self.memory_file = memory_file
        self._cache = {}  # Simple cache for frequent queries
        self._cache_timeout = 300  # Cache timeout in seconds
        self._last_cache_clear = time.time()
        
        from database import ConversationDatabase
        self.db = ConversationDatabase()
        # Migrate existing conversations if any
        if os.path.exists(self.memory_file):
            try:
                self.db.migrate_from_jsonl(self.memory_file)
                # Backup and rename the old file
                backup_name = f"{self.memory_file}.bak"
                os.rename(self.memory_file, backup_name)
                logger.info(f"Existing conversations migrated to database, old file backed up as {backup_name}")
            except Exception as e:
                logger.error(f"Error migrating conversations: {e}")
    
    def _clear_old_cache(self):
        """Clear expired cache entries."""
        current_time = time.time()
        if current_time - self._last_cache_clear > self._cache_timeout:
            self._cache.clear()
            self._last_cache_clear = current_time
    
    def add_exchange(self, user_input, model_response, user_feedback=None):
        """Add a conversation exchange to history and database."""
        exchange = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "model_response": model_response,
            "user_feedback": user_feedback
        }
        
        self.history.append(exchange)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Save to database
        try:
            self.db.add_exchange(user_input, model_response, user_feedback)
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            # Fallback to file if database fails
            with open(self.memory_file, 'a') as f:
                f.write(json.dumps(exchange) + "\n")
    
    def get_history_as_context(self, max_exchanges=3):
        """Return conversation history formatted as context."""
        context = ""
        for exchange in self.history[-max_exchanges:]:
            context += f"User: {exchange['user_input']}\n"
            context += f"Theta: {exchange['model_response']}\n"
        return context
    
    def get_training_pairs(self):
        """Get conversation pairs for training with caching."""
        self._clear_old_cache()
        cache_key = 'training_pairs'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            pairs = self.db.get_training_pairs()
            self._cache[cache_key] = pairs
            return pairs
        except Exception as e:
            logger.error(f"Error getting training pairs from database: {e}")
            # Fallback to memory
            pairs = []
            for exchange in self.history:
                pairs.append((exchange['user_input'], exchange['model_response']))
                if exchange.get('user_feedback') and exchange['user_feedback'] != exchange['model_response']:
                    pairs.append((exchange['user_input'], exchange['user_feedback']))
            return pairs

    def load_past_conversations(self, limit=100):
        """Load a limited number of past conversations from the database."""
        try:
            return self.db.get_training_pairs(limit)
        except Exception as e:
            logger.error(f"Error loading past conversations from database: {e}")
            # Fallback to file
            pairs = []
            try:
                if os.path.exists(self.memory_file):
                    with open(self.memory_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                exchange = json.loads(line)
                                pairs.append((
                                    exchange['user_input'],
                                    exchange.get('user_feedback', exchange['model_response'])
                                ))
                                if len(pairs) >= limit:
                                    break
            except Exception as file_error:
                logger.error(f"Error reading from file fallback: {file_error}")
            return pairs

class DictionaryDataset:
    """Provides dictionary and basic knowledge for Theta."""
    
    def __init__(self, include_wordnet=True):
        self.vocabulary = set()
        self.definitions = {}
        self.synonyms = {}
        self.conversation_templates = []
        self.ensure_nltk_resources()
        self.load_basic_vocabulary(include_wordnet)
        self.load_conversation_templates()
    
    def ensure_nltk_resources(self):
        """Ensure NLTK resources are downloaded."""
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            logger.info("Downloading NLTK words corpus...")
            nltk.download('words', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK WordNet corpus...")
            nltk.download('wordnet', quiet=True)
    
    def load_basic_vocabulary(self, include_wordnet):
        """Load basic vocabulary and definitions."""
        logger.info("Loading basic vocabulary...")
        self.vocabulary = set(w.lower() for w in words.words())
        
        if include_wordnet:
            logger.info("Loading WordNet definitions and synonyms...")
            common_words = ['hello', 'goodbye', 'yes', 'no', 'thanks', 'please',
                          'good', 'bad', 'help', 'time', 'day', 'name', 'person',
                          'computer', 'program', 'code', 'learn', 'think', 'know',
                          'understand', 'question', 'answer', 'model', 'ai', 'theta']
            
            for word in tqdm(common_words, desc="Loading common words"):
                synsets = wordnet.synsets(word)
                if synsets and synsets[0] is not None:
                    # Get definition
                    self.definitions[word] = synsets[0].definition()
                    
                    # Get synonyms, safely handling potential None values
                    synonyms = []
                    for synset in synsets:
                        if synset is not None and hasattr(synset, 'lemmas'):
                            for lemma in synset.lemmas():
                                if lemma is not None and hasattr(lemma, 'name'):
                                    syn_name = lemma.name().replace('_', ' ')
                                    if syn_name != word:
                                        synonyms.append(syn_name)
                    self.synonyms[word] = list(set(synonyms))
    
    def load_conversation_templates(self):
        """Load basic conversation templates."""
        self.conversation_templates = [
            # Greetings
            ("hello", "Hello! How can I help you today?"),
            ("hi", "Hi there! What can I do for you?"),
            ("hey", "Hey! What's on your mind?"),
            ("good morning", "Good morning! How are you today?"),
            ("good afternoon", "Good afternoon! How's your day going?"),
            ("good evening", "Good evening! How has your day been?"),
            
            # Identity
            ("what is your name", "My name is Theta, an AI assistant developed by Frostline Solutions."),
            ("who are you", "I'm Theta, an AI assistant created to help with coding tasks and answer questions."),
            ("who made you", "I was created by Dakota Fryberger at Frostline Solutions LLC."),
            
            # Capabilities
            ("what can you do", "I can help with coding tasks, answer questions, and have conversations like this one."),
            ("how do you work", "I'm a transformer-based neural network trained to understand and generate text."),
            
            # Farewells
            ("goodbye", "Goodbye! Feel free to chat with me again anytime."),
            ("bye", "Bye! Have a great day!"),
            ("see you later", "See you later! Looking forward to our next conversation."),
            
            # Feedback
            ("thank you", "You're welcome! Is there anything else I can help with?"),
            ("thanks", "You're welcome! Let me know if you need anything else."),
            
            # Basics
            ("how are you", "I'm functioning well, thank you for asking! How about you?"),
            ("what time is it", "I don't have access to real-time data like the current time."),
            ("what day is it", "I don't have access to real-time data like the current date."),
        ]
        
        # Add some coding-related templates
        coding_templates = [
            ("what is python", "Python is a high-level, interpreted programming language known for its readability and versatility."),
            ("what is javascript", "JavaScript is a programming language that enables interactive web pages and is an essential part of web applications."),
            ("what is html", "HTML (HyperText Markup Language) is the standard markup language for documents designed to be displayed in a web browser."),
            ("what is css", "CSS (Cascading Style Sheets) is a style sheet language used for describing the presentation of a document written in HTML."),
            ("what is a function", "A function is a block of organized, reusable code that is used to perform a single, related action."),
            ("what is a variable", "A variable is a storage location paired with an associated symbolic name, which contains some known or unknown quantity of information referred to as a value."),
            ("what is a loop", "A loop is a programming construct that repeats a group of commands."),
            ("what is an array", "An array is a data structure consisting of a collection of elements, each identified by at least one array index or key."),
            ("what is a class", "A class is a blueprint for creating objects (a particular data structure), providing initial values for state and implementations of behavior."),
        ]
        
        self.conversation_templates.extend(coding_templates)
    
    def get_definition(self, word):
        """Get definition for a word if available."""
        return self.definitions.get(word.lower(), f"I don't have a definition for '{word}'.")
    
    def get_synonyms(self, word):
        """Get synonyms for a word if available."""
        return self.synonyms.get(word.lower(), [])
    
    def get_training_pairs(self, num_pairs=50):
        """Get training pairs from templates."""
        if num_pairs >= len(self.conversation_templates):
            return self.conversation_templates
        else:
            return random.sample(self.conversation_templates, num_pairs)

class ThetaConversationTrainer:
    """Manages training and interaction with the Theta model."""
    
    def __init__(self, model, tokenizer, device, learning_rate=5e-6):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Initialize conversation memory and dictionary
        self.memory = ConversationMemory()
        self.dictionary = DictionaryDataset()
        
        # Training settings
        self.learning_rate = learning_rate
        self.accumulation_steps = 8  # Gradient accumulation for stability
        self.max_length = 128
    
    def generate_response(self, user_input, use_context=True, max_length=100, temperature=0.3, 
                        top_p=0.85, top_k=20, repetition_penalty=1.2):
        """Generate a response to user input."""
        
        # Check if this is a state capital question
        is_capital_question = any(x in user_input.lower() for x in ["capital", "capitol"])
        if is_capital_question and "what" in user_input.lower():
            # Extract state name from the question
            state = None
            for pattern in ["capital of ", "capitol of "]:
                if pattern in user_input.lower():
                    state = user_input.lower().split(pattern)[-1].strip("?. ")
                    break
            
            if state:
                # Lookup capital from the state_capitals.txt file
                try:
                    with open("data/state_capitals.txt", "r") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith(f"Q: What is the capital of {state.title()}?"):
                                capital = next(f).strip()[3:].strip(".")
                                return f"The capital of {state.title()} is {capital}."
                except Exception as e:
                    logger.warning(f"Could not validate capital answer: {e}")
        
        # Add conversation context if needed
        if use_context and self.memory.history:
            context = self.memory.get_history_as_context()
            prompt = f"{context}User: {user_input}\nTheta:"
        else:
            prompt = f"User: {user_input}\nTheta:"
            
        # Convert prompt to tensor
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, 
                                        max_length=self.tokenizer.model_max_length)
        
        # Move to the same device as the model
        input_ids = input_ids.to(self.device)
        
        try:
            # Determine if we should use beam search or sampling based on the type of input
            # For questions and commands, beam search produces more coherent, focused responses
            # For open-ended chat, sampling produces more creative, varied responses
            is_question = any(q in user_input.lower() for q in ["?", "what", "how", "when", "where", "why", "who", "which"])
            is_command = any(c in user_input.lower() for c in ["show", "tell", "list", "explain", "describe"])
            
            # Use more structured generation for questions/commands, more creative for chat
            with torch.no_grad():
                if is_question or is_command:
                    # More focused parameters for questions and commands using beam search
                    output_sequences = self.model.generate(
                        input_ids=input_ids,
                        max_length=input_ids.shape[1] + max_length,
                        min_length=20,  # Longer minimum response for complete sentences
                        num_beams=5,  # Use beam search for more coherent outputs
                        early_stopping=True,
                        no_repeat_ngram_size=3,  # Avoid repeating trigrams
                        length_penalty=1.5,  # Prefer longer responses
                        repetition_penalty=2.0,  # Strong repetition penalty
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # More creative parameters for general chat using sampling
                    output_sequences = self.model.generate(
                        input_ids=input_ids,
                        max_length=input_ids.shape[1] + max_length,
                        min_length=15,  # Ensure complete sentences
                        temperature=0.5,  # Lower temperature for more focused responses
                        top_k=20,  # More restrictive token filtering
                        top_p=0.8,  # More conservative sampling
                        repetition_penalty=2.0,  # Stronger repetition penalty to avoid loops
                        no_repeat_ngram_size=2,  # Avoid repeating bigrams
                        do_sample=True,  # Keep sampling enabled for chat
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

            # Decode the response
            generated_sequence = output_sequences[0]
            text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # Extract only the response part after the prompt
            response = text.split("Theta:")[-1].strip()
            
            # Clean up the response
            response = self._clean_response(response)
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def _clean_response(self, text):
        """Clean up the generated response."""
        # Remove any user prompts that might have been generated
        if "\nUser:" in text:
            text = text.split("\nUser:")[0]
        
        # Fix common generation issues
        text = text.replace("..", ".").replace("!!", "!").replace("??", "?")
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[.!?]+([^\s])', r'. \1', text)  # Add space after punctuation if missing
        
        # Fix sentence fragments and ensure complete sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        # Filter out very short or incomplete sentences
        filtered_sentences = []
        for sentence in sentences:
            # Skip very short fragments or those with excessive punctuation
            if len(sentence) < 3 or sentence.count('.') > 2 or sentence.count(',') > 3:
                continue
                
            # Ensure sentence ends with proper punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentence += "."
                
            filtered_sentences.append(sentence)
        
        # Ensure at least one complete sentence
        if not filtered_sentences:
            return "I'm thinking about this topic. Let me formulate a complete response."
        
        # Combine sentences back together
        cleaned_text = ' '.join(filtered_sentences)
        
        # Final cleanup to ensure proper spacing and capitalization
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Final whitespace normalization
        cleaned_text = cleaned_text.strip()
        
        # Capitalize first letter if needed
        if cleaned_text and cleaned_text[0].islower():
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
            
        return cleaned_text
    
    def train_on_batch(self, input_texts, target_texts):
        """Train the model on a batch of conversation exchanges."""
        self.model.train()
        
        # Process each pair one at a time for incremental learning
        total_loss = 0
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            # Prepare input
            prompt = f"User: {input_text}\nTheta: {target_text}"
            
            # Tokenize
            encodings = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Prepare inputs and labels for causal language modeling
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            # Calculate loss
            loss = outputs["loss"] / self.accumulation_steps
            total_loss += loss.item() * self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0 or i == len(input_texts) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return total_loss / len(input_texts)
    
    def learn_from_conversation(self, user_input, model_response, corrected_response=None):
        """Update the model based on the conversation exchange."""
        # If a correction was provided, train on the corrected version
        target = corrected_response if corrected_response else model_response
        
        # For capital questions, validate against known answers
        if "what is" in user_input.lower() and "capital" in user_input.lower():
            # Extract state name
            state_match = user_input.lower().split("capital of ")[-1].strip("?").strip()
            with open("data/state_capitals.txt", "r") as f:
                capitals_data = f.read()
                for line in capitals_data.split("\n"):
                    if f"capital of {state_match}" in line.lower() and "A:" in line:
                        correct_answer = line.split("A: ")[-1].strip()
                        if correct_answer != model_response:
                            target = correct_answer
                            corrected_response = correct_answer
        
        # Save to memory with any corrections
        self.memory.add_exchange(user_input, model_response, corrected_response)
        
        # Train on this exchange
        loss = self.train_on_batch([user_input], [target])
        
        # Also train on a small batch of past successful exchanges for stability
        past_conversations = self.memory.load_past_conversations(limit=5)
        if past_conversations:
            past_inputs, past_responses = zip(*past_conversations)
            self.train_on_batch(past_inputs, past_responses)
        
        # And train on some dictionary data for general knowledge
        dict_pairs = self.dictionary.get_training_pairs(num_pairs=3)
        if dict_pairs:
            dict_inputs, dict_responses = zip(*dict_pairs)
            self.train_on_batch(dict_inputs, dict_responses)
        
        return loss
    
    def save_model(self, output_dir):
        """Save the current model state."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save config if available
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
        # Also save some conversation memory statistics
        with open(os.path.join(output_dir, "memory_stats.txt"), "w") as f:
            f.write(f"Conversation exchanges: {len(self.memory.history)}\n")
            f.write(f"Memory file: {self.memory.memory_file}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def load_model(model_path):
    """Load the Theta model from a saved checkpoint."""
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(model_file):
        raise ValueError(f"Model file {model_file} does not exist")
    
    # Try to load config from file first
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            config = ThetaConfig(**config_dict)
            logger.info(f"Loaded config from {config_file}")
    else:
        # Fallback to default config
        config = ThetaConfig(
            vocab_size=50265,  # GPT-2 vocab size
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096
        )
        logger.info("Using default config")
    
    # Initialize model with config
    model = ThetaModel(config)
    
    # Load saved weights
    logger.info(f"Loading model from {model_file}")
    model.load_state_dict(torch.load(model_file))
    
    return model

def save_model_with_backup(model, tokenizer):
    """Save model with timestamp backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save backup
    backup_path = os.path.join(MODEL_DIR, "backups", f"theta_backup_{timestamp}")
    os.makedirs(backup_path, exist_ok=True)  # Create the full backup path
    
    # Save model state
    model_file = os.path.join(backup_path, "pytorch_model.bin")
    torch.save(model.state_dict(), model_file)
    
    # Save config if available
    config_dict = None
    if hasattr(model, 'config'):
        config_dict = model.config.__dict__ if hasattr(model.config, '__dict__') else {}
        os.makedirs(backup_path, exist_ok=True)
        with open(os.path.join(backup_path, "config.json"), "w") as f:
            json.dump(config_dict, f)
    
    # Save tokenizer
    tokenizer.save_pretrained(backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Save main model
    os.makedirs(MODEL_FILE, exist_ok=True)
    model_file = os.path.join(MODEL_FILE, "pytorch_model.bin")
    parent_dir = os.path.dirname(model_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(model.state_dict(), model_file)
    if config_dict is not None:
        with open(os.path.join(MODEL_FILE, "config.json"), "w") as f:
            json.dump(config_dict, f)
    tokenizer.save_pretrained(MODEL_FILE)
    logger.info(f"Model saved to {MODEL_FILE}")

def interactive_conversation_mode(args):
    """Run interactive conversation mode with real-time learning."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model_path)
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Initializing a new model")
        config = ThetaConfig()
        model = ThetaModel(config)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer
    trainer = ThetaConversationTrainer(model, tokenizer, device, learning_rate=args.learning_rate)
    
    # Conversation state tracking
    total_exchanges = 0
    save_frequency = args.save_every
    last_save_time = time.time()
    continuous_learning = args.continuous_learning
    
    # Default generation parameters that can be adjusted during conversation
    gen_params = {
        "temperature": 0.5,
        "max_length": 100,
        "min_length": 15,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 2.0,
        "no_repeat_ngram_size": 2
    }
    
    # Welcome message
    print("\n" + "="*50)
    print("Welcome to Theta AI Conversation Mode with Real-time Learning!")
    print("="*50)
    print("\nYou are now chatting with Theta. Type 'exit' to end the conversation.")
    print("Commands:")
    print("  /help - Show this help message")
    print("  /save - Save the current model state")
    print("  /learning on/off - Toggle continuous learning")
    print("  /correct - Correct the last response")
    print("  /stats - Show conversation statistics")
    print("  /params - Show current generation parameters")
    print("  /params set <name> <value> - Adjust generation parameters")
    print("  /exit - Exit conversation mode")
    print("\nTheta is learning from this conversation in real-time.")
    print("="*50 + "\n")
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Handle commands
        if user_input.lower() == "/exit" or user_input.lower() == "exit":
            print("\nSaving final model state...")
            save_model_with_backup(model, tokenizer)
            print("Goodbye! Thank you for chatting with Theta.")
            break
            
        elif user_input.lower() == "/help":
            print("\nCommands:")
            print("  /help - Show this help message")
            print("  /save - Save the current model state")
            print("  /learning on/off - Toggle continuous learning")
            print("  /correct - Correct the last response")
            print("  /stats - Show conversation statistics")
            print("  /exit - Exit conversation mode")
            continue
            
        elif user_input.lower() == "/save":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(args.output_dir, f"conversation_checkpoint_{timestamp}")
            trainer.save_model(save_path)
            print(f"\nModel saved to {save_path}")
            last_save_time = time.time()
            continue
            
        elif user_input.lower().startswith("/learning"):
            if "on" in user_input.lower():
                continuous_learning = True
                print("\nContinuous learning enabled. Theta will learn from this conversation.")
            elif "off" in user_input.lower():
                continuous_learning = False
                print("\nContinuous learning disabled. Theta will not update its model during this conversation.")
            else:
                print(f"\nContinuous learning is currently {'enabled' if continuous_learning else 'disabled'}")
            continue
            
        elif user_input.lower() == "/correct" and trainer.memory.history:
            last_exchange = trainer.memory.history[-1]
            print(f"\nLast question: {last_exchange['user_input']}")
            print(f"Theta's response: {last_exchange['model_response']}")
            correction = input("Enter the correct response: ")
            
            # Train on the correction
            loss = trainer.learn_from_conversation(
                last_exchange['user_input'], 
                last_exchange['model_response'],
                correction
            )
            print(f"\nTheta learned from your correction (loss: {loss:.4f})")
            continue
            
        elif user_input.lower() == "/stats":
            print("\nConversation Statistics:")
            print(f"  Total exchanges: {total_exchanges}")
            print(f"  Continuous learning: {'Enabled' if continuous_learning else 'Disabled'}")
            print(f"  Device: {device}")
            print(f"  Learning rate: {trainer.learning_rate}")
            if trainer.memory.history:
                print(f"  Memory size: {len(trainer.memory.history)} exchanges")
            continue
            
        elif user_input.lower().startswith("/params"):
            parts = user_input.lower().split()
            
            # Show current parameters
            if len(parts) == 1:
                print("\nCurrent Generation Parameters:")
                for param, value in gen_params.items():
                    print(f"  {param}: {value}")
                continue
                
            # Set a parameter
            elif len(parts) >= 4 and parts[1] == "set":
                param_name = parts[2]
                try:
                    param_value = float(parts[3])
                    
                    if param_name in gen_params:
                        # Apply some basic validation to prevent unusable values
                        if param_name == "temperature" and (param_value <= 0 or param_value > 2.0):
                            print("Temperature must be between 0.1 and 2.0")
                        elif param_name in ["top_p", "top_k", "repetition_penalty"] and param_value <= 0:
                            print(f"{param_name} must be greater than 0")
                        elif param_name in ["max_length", "min_length", "no_repeat_ngram_size"] and (param_value < 1 or not param_value.is_integer()):
                            print(f"{param_name} must be a positive integer")
                        else:
                            # Convert to integer if needed
                            if param_name in ["max_length", "min_length", "top_k", "no_repeat_ngram_size"]:
                                param_value = int(param_value)
                                
                            # Apply the change
                            gen_params[param_name] = param_value
                            print(f"Parameter {param_name} set to {param_value}")
                    else:
                        print(f"Unknown parameter: {param_name}")
                        print("Available parameters: " + ", ".join(gen_params.keys()))
                except ValueError:
                    print("Value must be a number")
                continue
        
        # Generate response using current parameters
        model_response = trainer.generate_response(
            user_input,
            max_length=gen_params['max_length'],
            temperature=gen_params['temperature'],
            top_p=gen_params['top_p'],
            top_k=gen_params['top_k'],
            repetition_penalty=gen_params['repetition_penalty']
        )
        print(f"\nTheta: {model_response}")
        
        # Learn from this exchange if continuous learning is enabled
        if continuous_learning:
            loss = trainer.learn_from_conversation(user_input, model_response)
            if args.show_learning_info:
                print(f"[Learning: loss={loss:.4f}]")
        
        # Increment exchange counter
        total_exchanges += 1
        
        # Periodic saving
        if (save_frequency > 0 and total_exchanges % save_frequency == 0) or \
           (args.save_minutes > 0 and (time.time() - last_save_time) > args.save_minutes * 60):
            save_path = os.path.join(args.output_dir, f"conversation_checkpoint_{total_exchanges}")
            trainer.save_model(save_path)
            print(f"\n[Model automatically saved to {save_path}]")
            last_save_time = time.time()

def main():
    parser = argparse.ArgumentParser(description="Theta AI Conversation Mode with Real-time Learning")
    
    parser.add_argument("--model_path", type=str, default=MODEL_FILE,
                      help="Path to the model file")
    parser.add_argument("--output_dir", type=str, default="theta_conversation_models", 
                      help="Directory to save model checkpoints")
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
                      help="Learning rate for real-time updates")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for model inference and training")
    parser.add_argument("--continuous_learning", action="store_true", 
                      help="Enable continuous learning from the start")
    parser.add_argument("--save_every", type=int, default=10, 
                      help="Save model every N conversation exchanges (0 to disable)")
    parser.add_argument("--save_minutes", type=int, default=15, 
                      help="Save model every N minutes (0 to disable)")
    parser.add_argument("--show_learning_info", action="store_true", 
                      help="Show learning information after each exchange")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start conversation mode
    interactive_conversation_mode(args)

if __name__ == "__main__":
    main()
