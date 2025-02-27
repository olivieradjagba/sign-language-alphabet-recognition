from typing import Literal

import numpy as np

import torch
import torch.nn as nn

from src.utils import scaled_dot_product_attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
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
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
        
    def forward(self, x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size() # Get the batch size and sequence length
        
        q = self.q_layer(y) # Apply the linear transformation
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) # Reshape the output to split the heads
        q = q.permute(0, 2, 1, 3) # Permute the dimensions to get the heads in the second dimension
        
        kv = self.kv_layer(x) # Apply the linear transformation
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim) # Reshape the output to split the heads
        kv = kv.permute(0, 2, 1, 3) # Permute the dimensions to get the heads in the second dimension
        k, v = kv.chunk(2, dim=-1) # Split the query, key and value for each head
        
        values, attention = scaled_dot_product_attention(q, k, v, mask) # Apply the scaled dot-product attention
        values = values.reshape(batch_size, sequence_length, self.d_model) # Reshape the values to concatenate the heads
        values = self.linear_layer(values) # Apply the linear transformation to get the final output
        
        return values


class LayerNormalization(nn.Module): # nn.LayerNorm can be used instead
    def __init__(self, params_shape:list[int], eps:float=1e-6):
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
    def __init__(self, d_model:int, hidden_size:int, drop_prob:float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, d_model)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, ffn_hidden_size:int, num_heads:int, drop_prob:float):
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
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model:int, ffn_hidden_size:int, num_heads:int, drop_prob:float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = nn.Dropout(drop_prob)
        self.mhca = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = FeedForwardNetwork(d_model, ffn_hidden_size, drop_prob)
        self.norm3 = LayerNormalization([d_model])
        self.dropout3 = nn.Dropout(drop_prob)
        
    def forward(self, x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        residual = y
        y = self.mha(y, mask)
        y = self.norm1(y + residual)
        y = self.dropout1(y)
        residual = y
        y = self.mhca(x, y)
        y = self.norm2(y + residual)
        y = self.dropout2(y)
        residual = y
        y = self.ffn(y)
        y = self.norm3(y + residual)
        y = self.dropout3(y)
        return y