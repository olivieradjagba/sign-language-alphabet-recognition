import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]
    scaled = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention_weights = F.softmax(scaled, dim=-1)
    values = attention_weights @ v
    return values, attention_weights