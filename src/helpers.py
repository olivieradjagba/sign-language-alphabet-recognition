import math
import torch
import torch.nn.functional as F
from torch import nn, optim

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]
    scaled = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention_weights = F.softmax(scaled, dim=-1)
    values = attention_weights @ v
    return values, attention_weights


class ViTLoss(nn.Module):
    def __init__(self, label_smoothing:float=0.0, **kwargs) -> None:
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(label_smoothing = label_smoothing, **kwargs)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        return self.loss_func(logits, labels)
    
    
class Scheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1):

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)#, verbose)
        
    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))