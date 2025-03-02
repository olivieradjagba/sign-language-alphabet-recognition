import torch
import torch.nn as nn

class PatchTokenizer(nn.Module):
    """
    Patch Tokenization Module for images embedding
    """
    def __init__(self,
                img_size: tuple[int, int],
                patch_size: int,
                n_channels: int,
                d_model: int=768):
        """
        Initialize the Patch Tokenization Module
        Args:
            img_size (tuple[int, int, int]): size of input (channels, height, width)
            patch_size (int): the side length of a square patch
            n_channels (int): number of channels in the input
            d_model (int): desired length of an output token
        """
        super().__init__()

        ## Defining Parameters
        H, W = img_size
        C = n_channels
        assert H % patch_size == 0, 'Height of image must be evenly divisible by patch size.'
        assert W % patch_size == 0, 'Width of image must be evenly divisible by patch size.'
        self.num_tokens = H * W // (patch_size ** 2)

        ## Defining Layers
        self.split = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0)
        self.project = nn.Linear((patch_size**2)*C, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.split(x).transpose(2,1)
        x = self.project(x)
        return x
    

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Module for images embedding
    """
    def __init__(self, patch_size:int, n_channels:int, d_model:int):
        """
        Initialize the Patch Embedding Module
        Args:
            patch_size (int): the side length of a square patch
            n_channels (int): number of channels in the input
            d_model (int): desired length of an output token
        """
        super().__init__()

        self.project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(-2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)

        return x

class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module
    """
    def __init__(self, max_sequence_length:int, d_model:int, learn:bool=False):
        """
        Initialize the Positional Encoding Module
        Args:
            max_sequence_length (int): maximum sequence length
            d_model (int): desired length of an output token
            learn (bool): whether to learn the positional encoding
        """
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token
        
        # Positional Encoding
        if learn:
            self.pe = nn.Parameter(torch.zeros(1, max_sequence_length, d_model))
        else:
            even_i = torch.arange(0, d_model, 2, dtype=torch.float)
            den = 10000 ** (even_i / d_model)
            pos = torch.arange(max_sequence_length, dtype=torch.float).unsqueeze(1)
            pe = torch.stack((torch.sin(pos / den), torch.cos(pos / den)), dim=-1)
            self.register_buffer('pe', pe.flatten(-2, -1).unsqueeze(0))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)
        # # Add positional encoding to embeddings
        x += self.pe

        return x
    