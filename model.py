"""
Theta AI Model: Core model definition and architecture.

This module defines the Theta model architecture for code understanding
and generation tasks. It builds on transformer architecture with
customizations for code-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import math
import copy


class ThetaConfig:
    """Configuration class for Theta model."""
    
    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_size: int = 768,         # Reduced from 1024 to prevent overfitting
        num_hidden_layers: int = 12,    # Reduced from 24 to prevent overfitting
        num_attention_heads: int = 12,  # Adjusted to match hidden size
        intermediate_size: int = 3072,  # Reduced from 4096 to prevent overfitting
        hidden_dropout_prob: float = 0.2,  # Increased from 0.1 for better regularization
        attention_probs_dropout_prob: float = 0.2,  # Increased from 0.1 for better regularization
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        base_model_type: str = "general",
        use_cache: bool = True,
        classifier_dropout: float = 0.2,  # Increased from 0.1 for better regularization
        model_type: str = "theta",
        rotary_dim: int = 32,
        use_rotary_embeddings: bool = True,
        activation_function: str = "gelu_new",
        weight_decay: float = 0.01,    # Added weight decay for regularization
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.base_model_type = base_model_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.weight_decay = weight_decay
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load configuration from a pretrained model."""
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=getattr(config, "hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=getattr(config, "attention_probs_dropout_prob", 0.1),
            max_position_embeddings=config.max_position_embeddings,
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-12),
            pad_token_id=config.pad_token_id,
            bos_token_id=getattr(config, "bos_token_id", 0),
            eos_token_id=getattr(config, "eos_token_id", 2),
        )
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to a directory."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, "theta_config.json")
        
        # Convert config to dictionary and save as JSON
        config_dict = self.to_dict()
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)


class ThetaModel(nn.Module):
    """
    Theta model for general AI capabilities including language understanding and generation.
    
    This model is designed as a from-scratch general-purpose AI with capabilities
    for understanding and generating text across multiple domains including code,
    general knowledge, reasoning, and creative content.
    """
    
    def __init__(self, config: ThetaConfig):
        super().__init__()
        self.config = config
        
        # Core components of the model
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = None
        
        # Use modern rotary positional embeddings if enabled
        if getattr(config, "use_rotary_embeddings", False):
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            self.rotary_dim = config.rotary_dim if hasattr(config, "rotary_dim") else 32
            self.rotary_emb = LlamaRotaryEmbedding(self.rotary_dim, config.max_position_embeddings)
        else:
            # Traditional positional embeddings
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Task-specific heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # Language modeling head
        
        # Domain-specific modules
        self.code_understanding = nn.Linear(config.hidden_size, config.hidden_size)  # For code-specific tasks
        self.knowledge_retrieval = nn.Linear(config.hidden_size, config.hidden_size)  # For fact retrieval
        self.reasoning_module = nn.Linear(config.hidden_size, config.hidden_size)  # For logical reasoning
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.classifier_dropout)
    
    def _init_weights(self, module):
        """Initialize the weights for the model."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer norm with ones and zeros
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, labels=None, return_dict=True, task=None):
        """Forward pass of the model.
        
        Args:
            input_ids: Token ids for input sequence
            attention_mask: Mask to avoid attention on padding tokens
            token_type_ids: Segment token indices for different segments (optional)
            position_ids: Indices for position embeddings (optional)
            labels: Labels for computing language modeling loss (optional)
            return_dict: Whether to return a dict or tuple (default: True)
            task: Specific task for the forward pass (optional)
            
        Returns:
            Dictionary with loss, logits, and optional hidden states and attentions
        """
        batch_size, seq_length = input_ids.size()
        
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        
        # Apply positional embeddings
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            # Generate position ids if not provided
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                
            # Generate rotary embeddings
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = None, None
            
            # Traditional positional embeddings
            if self.position_embeddings is not None:
                if position_ids is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.position_embeddings(position_ids)
                hidden_states = hidden_states + position_embeddings
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Convert attention mask from [0, 1] to [-10000, 0]
        if attention_mask is not None:
            # Expand attention mask to match the shape [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask from [0, 1] to [0.0, -10000.0]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Apply transformer layers
        all_hidden_states = (hidden_states,) if return_dict else None
        all_attentions = () if return_dict else None
        
        for layer in self.layers:
            if return_dict:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                output_attentions=return_dict,
                cos=cos,
                sin=sin
            )
            
            hidden_states = layer_outputs["hidden_states"]
            
            if return_dict and "attentions" in layer_outputs:
                all_attentions = all_attentions + (layer_outputs["attentions"],)
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if return_dict:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Calculate loss if labels are provided
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # Return results
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
                "last_hidden_state": hidden_states,
            }
        else:
            outputs = (logits, hidden_states)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=100, min_length=0,
                temperature=0.9, top_k=40, top_p=0.92, repetition_penalty=1.3,
                do_sample=True, num_return_sequences=1, pad_token_id=None, eos_token_id=None,
                task=None, **kwargs):
        """Generate text based on input_ids.
        
        Args:
            input_ids: Token ids for input sequence
            attention_mask: Mask to avoid attention on padding tokens
            max_length: Maximum length of the generated text
            min_length: Minimum length of the generated text
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            task: Specific task for generation
            
        Returns:
            Generated token sequences
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize output sequences with input sequences
        output_sequences = input_ids.clone()
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Loop until max length or all sequences finished
        while output_sequences.shape[1] < max_length and unfinished_sequences.any():
            # Get model predictions
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=output_sequences,
                    attention_mask=attention_mask,
                    task=task,
                    return_dict=True
                )
            
            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(output_sequences[i].tolist()):
                        if previous_token != pad_token_id:
                            next_token_logits[i, previous_token] /= repetition_penalty
            
            # Sample or use greedy decoding
            if do_sample:
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    # Apply softmax first to convert logits to probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a new tensor of -inf values with same shape as next_token_logits
                    filtered_logits = torch.ones_like(next_token_logits) * -float("inf")
                    
                    # For each batch item, keep only the tokens that pass the top-p filter
                    for batch_idx in range(batch_size):
                        indices_to_keep = sorted_indices[batch_idx][~sorted_indices_to_remove[batch_idx]]
                        filtered_logits[batch_idx, indices_to_keep] = next_token_logits[batch_idx, indices_to_keep]
                    
                    # Replace next_token_logits with the filtered version
                    next_token_logits = filtered_logits
                
                # Sample from the filtered distribution
                # Check for NaN or Inf values and replace with small values to ensure valid sampling
                is_inf = torch.isinf(next_token_logits)
                if is_inf.any():
                    # Set -inf values to a very negative but finite value
                    next_token_logits = torch.where(is_inf, torch.tensor(-1e10, device=next_token_logits.device), next_token_logits)
                
                # Apply softmax to get probabilities, adding a small epsilon to avoid numerical issues
                probs = F.softmax(next_token_logits, dim=-1) + 1e-10
                
                # Normalize to ensure probabilities sum to 1
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the probability distribution
                try:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                except RuntimeError:
                    # Fallback to argmax if multinomial fails (e.g., due to numerical issues)
                    next_tokens = torch.argmax(probs, dim=-1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update unfinished sequences mask
            if eos_token_id is not None:
                # Check if EOS token has been generated
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
            
            # Only keep unfinished sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Append next tokens to output sequences
            output_sequences = torch.cat([output_sequences, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                )
        
        # Return generated sequences
        return output_sequences


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: ThetaConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Ensure hidden size is divisible by number of heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f"Hidden size {self.hidden_size} not divisible by num heads {self.num_heads}")
        
        # Projection matrices
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Attention dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Flag for rotary embeddings
        self.use_rotary = getattr(config, "use_rotary_embeddings", False)
        self.rotary_dim = getattr(config, "rotary_dim", 0) if self.use_rotary else 0
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        """Reshape tensor for multi-head attention."""
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def apply_rotary_embeddings(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary positional embeddings to query and key tensors."""
        # Apply rotary embeddings only to the first rotary_dim dimensions
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        
        # Apply rotary embeddings formula: (q*cos + rotate(q)*sin)
        q_rot_cos = q_rot * cos
        q_rot_sin = q_rot.clone()
        q_rot_sin[..., 0::2] = -q_rot[..., 1::2]
        q_rot_sin[..., 1::2] = q_rot[..., 0::2]
        q_rot_sin = q_rot_sin * sin
        q_rot = q_rot_cos + q_rot_sin
        
        # Apply same to keys
        k_rot_cos = k_rot * cos
        k_rot_sin = k_rot.clone()
        k_rot_sin[..., 0::2] = -k_rot[..., 1::2]
        k_rot_sin[..., 1::2] = k_rot[..., 0::2]
        k_rot_sin = k_rot_sin * sin
        k_rot = k_rot_cos + k_rot_sin
        
        # Concatenate rotated and non-rotated parts
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        
        return q, k
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the self-attention layer."""
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project hidden states to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, seq_len, batch_size)
        key_states = self._shape(key_states, seq_len, batch_size)
        value_states = self._shape(value_states, seq_len, batch_size)
        
        # Handle cached key/value states for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        # Save current states for next iteration if caching
        if use_cache:
            current_key_value = (key_states, value_states)
        else:
            current_key_value = None
            
        # Apply rotary embeddings if enabled
        if self.use_rotary and cos is not None and sin is not None:
            query_states, key_states = self.apply_rotary_embeddings(query_states, key_states, cos, sin)
        
        # Get sequence length after potentially adding past states
        kv_seq_len = key_states.shape[2]
        
        # Compute attention scores: batch_size x num_heads x query_len x key_len
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add large negative values to masked positions
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(query_states)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Compute weighted sum of values using attention probabilities
        context_states = torch.matmul(attention_probs, value_states)
        
        # Reshape back to original dimensions
        context_states = context_states.transpose(1, 2).contiguous()
        context_states = context_states.view(batch_size, seq_len, self.hidden_size)
        
        # Apply output projection
        output = self.o_proj(context_states)
        
        outputs = {
            "hidden_states": output,
            "attentions": attention_probs if output_attentions else None,
            "past_key_value": current_key_value
        }
        
        return outputs


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: ThetaConfig):
        super().__init__()
        self.config = config
        
        self.activation_fn = self._get_activation_fn(getattr(config, "activation_function", "gelu"))
        
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def _get_activation_fn(self, activation_name):
        """Get activation function by name."""
        if activation_name == "gelu":
            return F.gelu
        elif activation_name == "gelu_new":
            return self._gelu_new
        elif activation_name == "relu":
            return F.relu
        elif activation_name == "silu":
            return F.silu
        else:
            return F.gelu
    
    def _gelu_new(self, x):
        """Implementation of GELUNew activation function."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward network."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """A single transformer layer combining self-attention and feed-forward networks."""
    
    def __init__(self, config: ThetaConfig):
        super().__init__()
        self.config = config
        
        # Self-attention mechanism
        self.attention = SelfAttention(config)
        
        # Layer normalization (pre-norm style, like in modern architectures)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None, 
        sin: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the transformer layer."""
        # Self-attention with pre-normalization
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cos=cos,
            sin=sin,
        )
        
        # Apply residual connection after attention
        hidden_states = residual + self.dropout(attention_outputs["hidden_states"])
        
        # Feed-forward network with pre-normalization
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        
        # Apply residual connection after feed-forward
        hidden_states = residual + hidden_states
        
        # Prepare outputs
        outputs = {
            "hidden_states": hidden_states,
            "attentions": attention_outputs["attentions"],
            "past_key_value": attention_outputs["past_key_value"],
        }
        
        return outputs
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_rotary_embedding(self, position_ids):
        """Get rotary embeddings for the given position IDs."""
        cos, sin = self.rotary_emb(position_ids)
        return cos.to(position_ids.device), sin.to(position_ids.device)
        
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load a pretrained Theta model."""
        config = ThetaConfig.from_pretrained(model_name_or_path)
        model = cls(config)
        # Load state dict if it's a Theta model
        # This would need custom implementation based on your saving format
        return model
    
    def generate(self, input_ids=None, attention_mask=None, max_length=100, min_length=10,
                temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2,
                do_sample=True, num_return_sequences=1, pad_token_id=None,
                eos_token_id=None, task=None, **kwargs):
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs to continue from
            attention_mask: Attention mask for input tokens
            max_length: Maximum length of the generated text
            min_length: Minimum length of the generated text
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to return
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            task: Specific task for generation
            
        Returns:
            Generated token IDs
        """
        device = next(self.parameters()).device
        batch_size = input_ids.shape[0]
        
        # Set pad_token_id and eos_token_id if not provided
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
            
        # Prepare for decoding
        cur_len = input_ids.shape[1]
        generated = input_ids.clone()
        
        # Get vocabulary size
        vocab_size = self.lm_head.out_features
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Main generation loop
        while cur_len < max_length:
            # Get model outputs
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    task=task,
                    return_dict=True
                )
                
                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature scaling
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Filter with top-k
                if top_k > 0:
                    # Keep only the top k tokens
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # Create a mask with zeros everywhere except for the top-k positions
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    # Scatter the top k values back to their original positions
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Filter with top-p (nucleus sampling)
                if top_p < 1.0 and do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')
                
                # Sample or do greedy decoding
                if do_sample:
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Add the next tokens to the generated sequence
                generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)
                
                # Update the attention mask for the new token
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1
                )
                
                # Check if all sequences have reached the EOS token
                if eos_token_id is not None:
                    if (generated[:, -1] == eos_token_id).all():
                        break
                
                # Update the current length
                cur_len = generated.shape[1]
                
                # Break if we've reached max_length
                if cur_len >= max_length:
                    break
                
        return generated
                
    def save_pretrained(self, save_directory: str):
        """Save the model to the specified directory."""
        # Save configuration
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        with open(os.path.join(save_directory, "theta_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
