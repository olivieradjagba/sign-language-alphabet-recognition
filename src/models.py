import torch
import torch.nn as nn
from torchvision import models

class SignLanguageCNN(nn.Module):
    """
    Convolutional Neural Network for Sign Language Recognition.
    """
    def __init__(self, num_classes:int):
        """
        Initialize the SignLanguageCNN.

        Args:
            num_classes (int): Number of classes in the dataset.
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
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class SignLanguageTL(nn.Module):
    def __init__(self, model:nn.Module, weights:models.WeightsEnum, num_classes:int):
        super().__init__()
        """
        Initialize the ResNet model for fine-tuning.

        Args:
            num_classes (int): Number of output classes.
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
            nn.Linear(196, num_classes)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x