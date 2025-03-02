import torch
import torch.nn as nn
from torchvision import models

from src.embedding import PatchEmbedding, PatchTokenizer, PositionalEncoding
from src.encoder import Encoder

class SignLanguageCNN(nn.Module):
    """
    Convolutional Neural Network for Sign Language Recognition.
    """
    def __init__(self, n_classes:int, n_channels:int=3):
        """
        Initialize the SignLanguageCNN.
        Args:
            n_classes (int): Number of classes in the dataset.
            n_channels (int): Number of channels in the input image.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25),
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class SignLanguageTL(nn.Module):
    """
    Transfer Learning model for Sign Language Recognition.
    """
    def __init__(self, n_classes:int, base_model:nn.Module, weights:models.WeightsEnum):
        super().__init__()
        """
        Initialize the ResNet model for fine-tuning.
        Args:
            n_classes (int): Number of output classes
            base_model (nn.Module): Pre-trained model
            weights (models.WeightsEnum): Weights to load
        """
        self.base_model = base_model(weights=weights)

        # Replace the fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, n_classes)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x
    
    
class SignLanguageViT(nn.Module):
    """
    Vision Transformer model for Sign Language Recognition.
    """
    def __init__(self,
                 n_classes:int, 
                 d_model:int=512,
                 ffn_hidden_size:int=2048,
                 img_size:tuple[int,int]=(224,224),
                 patch_size:int=16,
                 n_channels:int=3,
                 n_heads:int=8,
                 n_layers:int=6,
                 drop_prob:float=0.1,
                 learn_pos_enc:bool=False):
        """
        Initialize the Vision Transformer model.
        Args:    
            n_classes (int): Number of output classes
            d_model (int): Embedding dimension
            ffn_hidden_size (int): Hidden size of the feed forward network
            img_size (tuple[int,int]): Size of the input image
            patch_size (int): Size of the patches
            n_channels (int): Number of channels in the input image
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            drop_prob (float): Dropout probability
            learn_pos_enc (bool): Whether to learn the positional encoding
        """
        super().__init__()

        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        n_patches = (img_size[0] * img_size[1]) // (patch_size ** 2)
        max_seq_length = n_patches + 1

        # self.embedding = PatchEmbedding(patch_size, n_channels, d_model)
        self.embedding = PatchTokenizer(img_size, patch_size, n_channels, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model, learn=learn_pos_enc)
        self.encoder = Encoder(d_model, ffn_hidden_size, n_heads, n_layers, drop_prob)
        
        # Classifier
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.classifier(x[:,0])

        return x