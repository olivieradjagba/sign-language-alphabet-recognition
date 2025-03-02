import torch
import torch.nn as nn

from src.layers import MultiHeadAttention, LayerNormalization, FeedForwardNetwork


class EncoderBlock(nn.Module):
    """
    Transformer Encoder block.
    """
    def __init__(self, d_model:int, ffn_hidden_size:int, num_heads:int, drop_prob:float):
        """
        Initialize the Encoder block.
        Args:
            d_model (int): Embedding dimension
            ffn_hidden_size (int): Hidden size of the feed forward network
            num_heads (int): Number of attention heads
            drop_prob (float): Dropout probability
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = FeedForwardNetwork(d_model, ffn_hidden_size, drop_prob)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(drop_prob)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mha(x)
        x = self.norm1(x + residual)
        x = self.dropout1(x)
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        
        return x

class Encoder(nn.Module):
    """
    Transformer Encoder.
    """
    def __init__(self, d_model:int, ffn_hidden_size:int, num_heads:int, num_layers:int, drop_prob:float):
        """
        Initialize the Encoder.
        Args:
            d_model (int): Embedding dimension
            ffn_hidden_size (int): Hidden size of the feed forward network
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            drop_prob (float): Dropout probability
        """
        super().__init__()
        self.layers = nn.Sequential(*[EncoderBlock(d_model, ffn_hidden_size, num_heads, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x