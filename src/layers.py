import torch
import torch.nn as nn

from src.utils import scaled_dot_product_attention
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.
    """
    def __init__(self, d_model:int, num_heads:int):
        """
        Initialize the Multi-Head Attention layer.
        Args:
            d_model (int): Embedding dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
        
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size() # Get the batch size and sequence length
        qkv = self.qkv_layer(x) # Apply the linear transformation
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)# # Reshape the output to split the heads
        qkv = qkv.permute(0, 2, 1, 3) # Permute the dimensions to get the heads in the second dimension
        q, k, v = qkv.chunk(3, dim=-1) # Split the query, key and value for each head
        values, attention = scaled_dot_product_attention(q, k, v, mask) # Apply the scaled dot-product attention
        values = values.reshape(batch_size, sequence_length, self.d_model) # Reshape the values to concatenate the heads
        values = self.linear_layer(values) # Apply the linear transformation to get the final output
        
        return values


class LayerNormalization(nn.Module): # nn.LayerNorm can be used instead
    """
    Layer Normalization layer.
    """
    def __init__(self, params_shape:list[int], eps:float=1e-6):
        """
        Initialize the Layer Normalization layer.
        Args:
            params_shape (list[int]): Shape of the parameters
            eps (float): Epsilon value
        """
        super().__init__()
        self.params_shape = params_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(params_shape))
        self.beta = nn.Parameter(torch.zeros(params_shape))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        dims = [-(i + 1) for i in range(len(self.params_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True) + self.eps
        y = self.gamma * (x - mean) / std + self.beta
        return y
    

class FeedForwardNetwork(nn.Module):
    """
    Point-wise Feed Forward Neural Network layer.
    """
    def __init__(self, d_model:int, hidden_size:int, drop_prob:float):
        """
        Initialize the Feed Forward Network layer.
        Args:
            d_model (int): Embedding dimension
            hidden_size (int): Hidden size of the feed forward network
            drop_prob (float): Dropout probability
        """ 
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, d_model)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    