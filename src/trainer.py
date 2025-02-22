# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

class SignLanguageModelTrainer:
    def __init__(self,
                 model:nn.Module,
                 classes:list[str],
                 init_model:bool=True,
                 save_path:str=None,
                 device:torch.device=torch.device("cpu")):
        """
        Initialize the ResNet model for fine-tuning.

        Args:
            model (nn.Module): The ResNet model.
            save_path (str): Path to save the model.
            device (torch.device): Device to run the model on.
        """
        self.model = model.to(device)
        if init_model:
            self.__init_model()
        self.classes = classes
        self.save_path = save_path
        self.device = device

    def __init_model(self) -> None:
        # Initialize the model with Xavier initialization
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def train(self,
              train:DataLoader,
              val:DataLoader,
              train_criterion:nn.modules.loss._Loss,
              val_criterion:nn.modules.loss._Loss,
              optimizer:optim.Optimizer,
              scheduler:optim.lr_scheduler._LRScheduler=None,
              epochs:int=10,
              patience:int=5,
              print_every:int=1,
              save_best:bool=False
    ) -> tuple[list[float], list[float]]:
        """
        Train the model.

        Args:
            train (DataLoader): Training data loader.
            val (DataLoader): Validation data loader.
            criterion (_Loss): Loss function.
            epochs (int): Number of training epochs.
            save_best (bool): Save the best model.
            
        Returns:
            tuple[list[float], list[float]]: Training and validation losses.
        """
        assert save_best and self.save_path or not save_best, "Save path must be provided if save_best is True."

        train_losses, val_losses = [], []
        best_val_loss = float('inf')  # Track the best validation loss
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = train_criterion(outputs, labels)
                # print(loss.item())
                # Backpropagation
                loss.backward()
                # Gradient clipping
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Update weights
                optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(train)
            val_loss = self.evaluate(val, val_criterion, is_test=False)
            
            # Save losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch == 0 or (epoch+1) % print_every == 0:
                print(f"Epoch [{epoch+1:>3}/{epochs}] Loss: {train_loss:.5f} | Val loss: {val_loss:.5f}",
                      f"| LR: {scheduler.get_last_lr()[0]:.6f}" if scheduler is not None else "",
                      f"-- Best model {'saved' if save_best else ''}" if val_loss < best_val_loss
                      else f"-- Best val loss: {best_val_loss:.5f} | Patience: {early_stopping_counter+1}/{patience}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                if save_best:
                    self.save_model(self.save_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping after epoch {epoch+1}")
                    break
                
            # Update learning rate
            if scheduler is not None:
                scheduler.step()

        if save_best:
            self.load_model()
        
        return train_losses, val_losses

    def evaluate(self, val:DataLoader, criterion:nn.modules.loss._Loss=None, is_test:bool=False) -> float | tuple[float, list[int], list[int]]:
        """
        Evaluate the model on validation data.

        Args:
            val (DataLoader): Validation data loader.
        """
        assert not is_test and criterion or is_test, "Criterion must be provided for evaluation."
        
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                if is_test:
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                else:
                    val_loss += criterion(outputs, labels).item()

        # print(f"Validation Accuracy: {100 * correct / total}%")
        return (100 * correct / total, y_true, y_pred) if is_test else val_loss / len(val)
    
    def predict(self, input:torch.Tensor) -> str:
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            _, pred_idx = torch.max(output, 1).item()
            return self.classes[pred_idx]
        
    def confusion_matrix(self, y_true:torch.Tensor, y_pred:torch.Tensor, num_classes:int) -> torch.Tensor:
        """
        Computes the confusion matrix for a multi-class classification problem.

        Args:
            y_true (torch.Tensor): Ground truth labels (1D tensor of size N).
            y_pred (torch.Tensor): Predicted labels (1D tensor of size N).
            num_classes (int): Number of classes.

        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes, num_classes).
        """
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        return cm

    def plot_confusion_matrix(self,
                              y_true:torch.Tensor,
                              y_pred:torch.Tensor,
                              classes:list[str],
                              normalize:bool=False,
                              figsize:tuple[int,int]=(12,10),
                              **kwargs) -> None:
        """
        Plots the confusion matrix as a heatmap.

        Args:
            y_true (torch.Tensor): Ground truth labels (1D tensor of size N).
            y_pred (torch.Tensor): Predicted labels (1D tensor of size N).
            classes (list): List of class names corresponding to the indices in the matrix.
            normalize (bool): Whether to normalize the confusion matrix to percentages.
            figsize (tuple): Figure size.
            **kwargs: Additional arguments to pass to seaborn.
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm / cm.sum(1, keepdims=True) * 100

        fig, ax = plt.subplots(figsize=figsize)
        cm_df = pd.DataFrame(cm, classes, classes)
        sns.heatmap(cm_df, annot=True, **kwargs)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        plt.show()
    
    def load_model(self, path:str=None) -> None:
        """
        Load the model from a file. If no path is provided, the model is loaded from the save path.
        """
        assert path or self.save_path, "Save path must be provided to load the model."
        
        path = path or self.save_path
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
    def save_model(self, path:str=None) -> None:
        """
        Save the model to a file.
        """
        assert path or self.save_path, "Save path must be provided to save the model."
        
        torch.save(self.model.state_dict(), path or self.save_path)
        
    def plot_losses(self, train_losses:list[float], val_losses:list[float]) -> None:
        """
        Plot the training and validation losses.
        """

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        