import torch
import torch.nn as nn

class PatchTokenization(nn.Module):
    def __init__(self,
                img_size: tuple[int, int, int]=(1, 28, 100),
                patch_size: int=50,
                token_len: int=768):

        """ Patch Tokenization Module
            Args:
                img_size (tuple[int, int, int]): size of input (channels, height, width)
                patch_size (int): the side length of a square patch
                token_len (int): desired length of an output token
        """
        super().__init__()

        ## Defining Parameters
        # self.img_size = img_size
        C, H, W = img_size
        # self.patch_size = patch_size
        # self.token_len = token_len
        assert H % patch_size == 0, 'Height of image must be evenly divisible by patch size.'
        assert W % patch_size == 0, 'Width of image must be evenly divisible by patch size.'
        self.num_tokens = H * W // (patch_size ** 2)

        ## Defining Layers
        self.spatial_encoding = nn.Parameter(torch.randn(patch_size**2, C))
        self.split = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0)
        self.project = nn.Linear((patch_size**2)*C, token_len)

    def forward(self, x):
        x = self.split(x).transpose(2,1)
        x = self.project(x)
        return x