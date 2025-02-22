import torch
import torch.nn as nn
from torchvision import models

from src.embedding import PatchEmbedding, PatchTokenizer, PositionalEncoding
from src.encoder import Encoder

class SignLanguageCNN(nn.Module):
    """
    Convolutional Neural Network for Sign Language Recognition.
    """
    def __init__(self, n_classes:int):
        """
        Initialize the SignLanguageCNN.

        Args:
            n_classes (int): Number of classes in the dataset.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
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
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
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
    def __init__(self, n_classes:int, model:nn.Module, weights:models.WeightsEnum):
        super().__init__()
        """
        Initialize the ResNet model for fine-tuning.

        Args:
            n_classes (int): Number of output classes.
        """
        self.model = model(weights=weights)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the parameters of the last few layers for fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 196),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(196, n_classes)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
    
    
class SignLanguageViT(nn.Module):
    def __init__(self,
                 n_classes:int, 
                 d_model:int,
                 ffn_hidden_size:int,
                 img_size:tuple[int,int]=(224,224),
                 patch_size:int=16,
                 n_channels:int=3,
                 n_heads:int=12,
                 n_layers:int=12,
                 drop_prob:float=0.1,
                 learnable_pos_enc:bool=False):
        super().__init__()

        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        n_patches = (img_size[0] * img_size[1]) // (patch_size ** 2)
        max_seq_length = n_patches + 1

        # self.patch_embedding = PatchEmbedding(patch_size, n_channels, d_model)
        self.patch_embedding = PatchTokenizer(img_size, patch_size, n_channels, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model, learnable=learnable_pos_enc)
        self.encoder = Encoder(d_model, ffn_hidden_size, n_heads, n_layers, drop_prob)
        # self.classifier = ViTClassifier(d_model, n_classes)
        
        # Classifier
        self.classifier = nn.Sequential(
            # nn.Linear(d_model, d_model // 4),
            # nn.GELU(),
            # nn.Dropout(drop_prob),
            # nn.Linear(d_model // 4, n_classes),
            nn.Linear(d_model, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.classifier(x[:,0])

        return x