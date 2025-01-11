import torch
import torch.nn as nn

from transformer_layers import DecoderLayer

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs) -> torch.Tensor:
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y
    

class Decoder(nn.Module):
    def __init__(self, d_model:int, ffn_hidden_size:int, num_heads:int, num_layers:int, num_patches:int, drop_prob:float):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden_size, num_heads, num_patches, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        x = self.layers(x, y, mask)
        return x