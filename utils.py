"""
Utility functions for the Theta AI model.

This module contains helper functions for data processing, tokenization,
evaluation metrics, and other utilities needed for training and using the Theta model.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader


def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


class CodeDataset(Dataset):
    """Dataset class for code data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        mode: str = "train",
    ):
        """
        Initialize CodeDataset.
        
        Args:
            data_path: Path to the dataset file or directory
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            mode: Dataset mode ('train', 'eval', or 'test')
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.examples = self.load_examples()
        
    def load_examples(self) -> List[Dict]:
        """Load examples from data path."""
        # This is a placeholder implementation
        # In a real scenario, you would load and parse your code dataset here
        examples = []
        
        # Example placeholder data
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        code_snippet = line.strip()
                        if code_snippet:
                            examples.append({"code": code_snippet})
                    except Exception as e:
                        print(f"Error processing line: {e}")
        
        print(f"Loaded {len(examples)} examples from {self.data_path}")
        return examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        example = self.examples[idx]
        code = example["code"]
        
        # Tokenize the code
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to appropriate format and remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # For masked language modeling (MLM)
        if self.mode == "train":
            # Create masked inputs for training
            inputs, labels = self.mask_tokens(item["input_ids"].clone())
            item["input_ids"] = inputs
            item["labels"] = labels
            
        return item
    
    def mask_tokens(
        self, inputs: torch.Tensor, mlm_probability: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        
        Args:
            inputs: The input tokens
            mlm_probability: Probability of masking a token
            
        Returns:
            Tuple of masked inputs and corresponding labels
        """
        labels = inputs.clone()
        
        # Create a mask array for tokens to be masked
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels for non-masked tokens to -100 (ignored in loss computation)
        labels[~masked_indices] = -100
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10% of the time) keep the masked input tokens unchanged
        return inputs, labels


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, task: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model predictions.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task: The task being evaluated
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    metrics = {}
    
    if task == "language_detection":
        # For classification tasks
        preds = np.argmax(predictions, axis=1)
        metrics["accuracy"] = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        
    elif task in ["understanding", "generation"]:
        # For code generation tasks, we need specialized metrics
        # This is a placeholder - you would implement code-specific metrics here
        metrics["perplexity"] = torch.exp(torch.tensor(predictions)).item()
        
        # Example placeholder for code correctness metrics
        metrics["code_accuracy"] = 0.0  # Would be calculated based on task-specific evaluation
        
    return metrics


def prepare_code_sample(code: str, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Prepare a code sample for inference.
    
    Args:
        code: Input code string
        tokenizer: Tokenizer for processing the code
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs
    """
    # Tokenize the code
    encoding = tokenizer(
        code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return encoding


def format_code_suggestion(generated_ids: torch.Tensor, tokenizer: PreTrainedTokenizer) -> str:
    """
    Format the model's generated output as readable code.
    
    Args:
        generated_ids: Model's generated token IDs
        tokenizer: Tokenizer for decoding the IDs
        
    Returns:
        Formatted code as a string
    """
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code
