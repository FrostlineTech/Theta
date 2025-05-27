"""
Inference script for the Theta AI model.

This script provides inference functionality for the trained Theta model,
allowing it to be used for code completion, understanding, and generation.
"""

import os
import argparse
import torch
import json
from transformers import AutoTokenizer

# Import local modules
from model import ThetaConfig, ThetaCodeModel
from utils import prepare_code_sample, format_code_suggestion


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with the Theta AI model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory")
    
    # Input arguments
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to input code file for inference")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Direct code text input for inference")
    
    # Task arguments
    parser.add_argument("--task", type=str, default="generation",
                        choices=["understanding", "generation", "language_detection"],
                        help="Task to perform inference for")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the model output")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu). If None, will use cuda if available.")
    
    # Other arguments
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of return sequences")
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path, device):
    """Load the model and tokenizer from the specified path."""
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Attempting to load base tokenizer...")
        # Try to determine the base model from config
        config_path = os.path.join(model_path, "theta_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            base_model_type = config_dict.get("base_model_type", "codebert")
            
            if base_model_type == "codebert":
                tokenizer_name = "microsoft/codebert-base"
            elif base_model_type == "codegpt":
                tokenizer_name = "microsoft/CodeGPT-small-py"
            elif base_model_type == "codet5":
                tokenizer_name = "Salesforce/codet5-small"
            else:
                tokenizer_name = base_model_type
                
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Default to CodeBERT tokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Load model - custom loading for Theta model
    try:
        # Check if this is our custom Theta model by looking for theta_config.json
        config_path = os.path.join(model_path, "theta_config.json")
        if os.path.exists(config_path):
            # Load our custom Theta configuration
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ThetaConfig(**config_dict)
            
            # Create the model with this config
            model = ThetaCodeModel(config)
            
            # Load the model weights
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
            else:
                # Try with model.safetensors
                weights_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(weights_path):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(weights_path)
                        model.load_state_dict(state_dict)
                    except ImportError:
                        print("Warning: safetensors package not found, falling back to torch.load")
                        # Fall back to torch.load
                        state_dict = torch.load(weights_path, map_location=device)
                        model.load_state_dict(state_dict)
                else:
                    raise FileNotFoundError(f"No model weights found in {model_path}")
        else:
            # Not a Theta model, try standard loading
            model = ThetaCodeModel.from_pretrained(model_path)
            
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return model, tokenizer, device


def generate_code(model, tokenizer, input_code, task, device, max_length=512, 
                  temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1):
    """Generate code using the Theta model."""
    # Prepare input
    inputs = prepare_code_sample(input_code, tokenizer, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        if task == "generation":
            # For generation, we use the model's forward pass to get logits
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task=task
            )
            
            # Get logits for next token prediction
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][:, -1, None] <= next_token_logits
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Combine with input sequence
            generated_ids = torch.cat([inputs["input_ids"], next_token.unsqueeze(-1)], dim=-1)
            
            # Continue generating tokens up to max_length
            for _ in range(max_length - inputs["input_ids"].shape[1] - 1):
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    task=task
                )
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][:, -1, None] <= next_token_logits
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Stop if end of sequence token is generated
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Combine with previous sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Format the generated code
            generated_code = format_code_suggestion(generated_ids, tokenizer)
            
        elif task == "understanding":
            # For understanding, we want to extract features for the code
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task=task
            )
            
            # In a real scenario, you'd use these features for understanding tasks
            # For demonstration, we'll just provide a simple code analysis
            generated_code = "Code Analysis:\n"
            generated_code += f"- Code length: {inputs['input_ids'].shape[1]} tokens\n"
            generated_code += "- Code understanding features extracted successfully\n"
            
        elif task == "language_detection":
            # Detect programming language
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task=task
            )
            
            # Get language prediction
            language_id = torch.argmax(outputs["logits"], dim=1).item()
            
            # Map language ID to language name (this is an example mapping)
            language_map = {
                0: "Python",
                1: "JavaScript",
                2: "Java",
                3: "C++",
                4: "C#",
                5: "PHP",
                6: "Ruby",
                7: "Go",
                8: "Swift",
                9: "Other"
            }
            
            detected_language = language_map.get(language_id, "Unknown")
            generated_code = f"Detected Programming Language: {detected_language}"
        
        else:
            raise ValueError(f"Task {task} is not supported")
    
    return generated_code


def run_inference(args):
    """Run inference with the Theta model."""
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Get input code
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_code = f.read()
    elif args.input_text:
        input_code = args.input_text
    else:
        input_code = input("Enter code: ")
    
    # Generate code
    generated_code = generate_code(
        model=model,
        tokenizer=tokenizer,
        input_code=input_code,
        task=args.task,
        device=device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences
    )
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        print(f"Output saved to {args.output_file}")
    else:
        print("\nGenerated Output:")
        print("-" * 40)
        print(generated_code)
        print("-" * 40)
    
    return generated_code


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
