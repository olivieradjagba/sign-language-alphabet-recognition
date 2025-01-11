import torch
import torch.nn as nn

from transformer_layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model:int, ffn_hidden_size:int, num_heads:int, num_layers:int, drop_prob:float):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden_size, num_heads, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x