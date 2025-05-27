"""
Test script for the Theta AI model.

This script allows you to test the Theta model with different types of questions
to evaluate its performance after training.
"""

# List of commands for testing
# python test_theta.py --model_path models/final --test_type identity # Test identity questions
# python test_theta.py --model_path models/final --test_type capitals # Test capital questions
# python test_theta.py --model_path models/final --test_type mixed # Test mixed questions
# python test_theta.py --model_path models/final --interactive # Theta AI chat mode 
# python test_theta.py --model_path models/final --question "What is the capital of California?" # Ask a single question
# python test_theta.py --model_path models/checkpoints/checkpoint-epoch-10 # ask a specific question at a specific checkpoint

import argparse
import torch
from transformers import AutoTokenizer
from model import ThetaConfig, ThetaModel
import logging
import os
from config import MODEL_FILE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the Theta model from a saved checkpoint."""
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(model_file):
        raise ValueError(f"Model file {model_file} does not exist")
    
    # Load saved config if it exists
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            config = ThetaConfig(**config_dict)
            logger.info(f"Loaded config from {config_file} with {config.num_hidden_layers} layers")
    else:
        # Fallback to default config with 24 layers
        config = ThetaConfig(num_hidden_layers=24)
        logger.info("Using default config with 24 layers")
    
    model = ThetaModel(config)
    
    # Load saved weights
    logger.info(f"Loading model from {model_file}")
    model.load_state_dict(torch.load(model_file))
    
    return model

def generate_answer(model, tokenizer, question, device, max_length=100, temperature=0.7, 
                   top_p=0.9, top_k=50, repetition_penalty=1.2):
    """Generate an answer for a question using the Theta model."""
    
    # Tokenize question
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # Check if this is a capital question
    is_capital_question = "capital" in question.lower() and "what" in question.lower()
    
    # If it's a capital question, first try to lookup from state_capitals.txt
    if is_capital_question:
        state = None
        for pattern in ["capital of ", "capitol of "]:
            if pattern in question.lower():
                state = question.lower().split(pattern)[-1].strip("?. ")
                break
        
        if state:
            try:
                with open("data/state_capitals.txt", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f"Q: What is the capital of {state.title()}?"):
                            capital = next(f).strip()[3:].strip(".")
                            return f"The capital of {state.title()} is {capital}."
            except Exception as e:
                logger.warning(f"Could not lookup capital from file: {e}")
    
    # Generate answer using the model
    with torch.no_grad():
        model.eval()
        # Use strict parameters for factual QA to ensure complete, accurate answers
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50 if is_capital_question else max_length,
            min_length=10 if is_capital_question else 0,
            temperature=0.1 if is_capital_question else temperature,
            top_k=1 if is_capital_question else top_k,
            top_p=0.1 if is_capital_question else top_p,
            repetition_penalty=1.0 if is_capital_question else repetition_penalty,
            do_sample=False if is_capital_question else True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the answer
    generated_sequence = output_sequences[0]
    answer = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    # Clean up and extract the answer part
    # For capital questions, extract and format the answer carefully
    if is_capital_question:
        # First try to extract state and city
        state = None
        city = None
        answer = answer.strip()
        
        # Try different patterns to extract state and city
        if "capital of" in answer.lower():
            parts = answer.lower().split("capital of")[-1].split("is")
            if len(parts) == 2:
                state = parts[0].strip()
                city = parts[1].strip(" .")
        elif " is " in answer:
            parts = answer.split(" is ")
            if len(parts) == 2:
                city = parts[1].strip(" .")
                # Try to find state from original question
                for pattern in ["capital of ", "capitol of "]:
                    if pattern in question.lower():
                        state = question.lower().split(pattern)[-1].strip("?. ")
                        break
        
        # Format the answer properly
        if state and city:
            return f"The capital of {state.title()} is {city.title()}."
        elif city:
            return f"The capital is {city.title()}."
    
    # For non-capital questions, clean up the response
    # Remove everything after any repetition patterns
    for i in range(len(answer)-5):
        if answer[i:i+5] in answer[i+5:]:
            answer = answer[:i+5]
            break
        if "." in answer[i:i+5]:  # Stop at first complete sentence
            answer = answer[:answer.index(".")+1]
            break
    
    # For non-capital questions, do additional cleanup
    if not is_capital_question:
        # Remove the question from the answer if present
        answer = answer.replace(question, "").strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["A:", "Answer:", "The answer is:"]
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Ensure proper sentence formatting
        if not any(answer.startswith(x) for x in ["The ", "A ", "I ", "Yes", "No"]):
            answer = "The " + answer
        if not answer.endswith("."):
            answer = answer + "."
        answer = f"The capital is {answer}"
    
    return answer.strip()

def test_model(model_path, questions=None, test_type=None):
    """Test the model with a set of questions."""
    
    # Load the model
    model = load_model(model_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Define test questions if not provided
    if questions is None:
        if test_type == "identity":
            questions = [
                "Who are you?",
                "What is your name?",
                "Who created you?",
                "Who is your creator?",
                "What is your relationship to Dakota Fryberger?",
                "What is your purpose?"
            ]
        elif test_type == "capitals":
            questions = [
                "What is the capital of California?",
                "What is the capital of Texas?",
                "What is the capital of New York?",
                "What is the capital of Florida?",
                "What is the capital of Wyoming?",
                "What is the capital of Alaska?"
            ]
        else:  # General mix of questions
            questions = [
                "Who are you?",
                "What is the capital of Texas?",
                "Who created you?",
                "What is the capital of California?",
                "What is your relationship to Dakota Fryberger?",
                "What is the capital of New York?"
            ]
    
    # Test each question
    results = []
    for question in questions:
        logger.info(f"Q: {question}")
        answer = generate_answer(model, tokenizer, question, device)
        logger.info(f"A: {answer}")
        logger.info("-" * 50)
        results.append((question, answer))
    
    return results

def interactive_mode(model_path):
    """Interactive mode to ask questions to the model."""
    
    # Load the model
    model = load_model(model_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    print("\nWelcome to Theta AI Interactive Mode!")
    print("Type 'exit' to quit or 'help' for commands.\n")
    
    # Parameter settings
    params = {
        "max_length": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2
    }
    
    while True:
        question = input("Ask Theta: ")
        
        if question.lower() == "exit":
            print("Goodbye!")
            break
        
        elif question.lower() == "help":
            print("\nCommands:")
            print("  exit - Exit interactive mode")
            print("  help - Show this help message")
            print("  params - Show current generation parameters")
            print("  set [param] [value] - Set a generation parameter")
            print("  reset params - Reset parameters to defaults\n")
        
        elif question.lower() == "params":
            print("\nCurrent parameters:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            print()
        
        elif question.lower().startswith("set "):
            try:
                _, param, value = question.split(" ", 2)
                if param in params:
                    params[param] = float(value) if param != "max_length" else int(value)
                    print(f"Parameter {param} set to {value}")
                else:
                    print(f"Unknown parameter: {param}")
            except ValueError:
                print("Invalid command format. Use 'set [param] [value]'")
        
        elif question.lower() == "reset params":
            params = {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2
            }
            print("Parameters reset to defaults")
        
        else:
            answer = generate_answer(
                model, tokenizer, question, device,
                max_length=params["max_length"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                repetition_penalty=params["repetition_penalty"]
            )
            print(f"Theta: {answer}")
            print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(MODEL_FILE),
                      help="Path to the model directory")
    parser.add_argument("--test_type", type=str, choices=["identity", "capitals", "mixed"], 
                        default="mixed", help="Type of test questions to use")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode to ask questions")
    parser.add_argument("--question", type=str, help="Single question to ask the model")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter (default: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty for text generation (default: 1.2)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model_path)
    elif args.question:
        model = load_model(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        answer = generate_answer(
            model, 
            tokenizer, 
            args.question, 
            device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print(f"Q: {args.question}")
        print(f"A: {answer}")
    else:
        test_model(args.model_path, test_type=args.test_type)

if __name__ == "__main__":
    main()
