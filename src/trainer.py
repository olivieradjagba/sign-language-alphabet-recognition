import os
from typing import Literal
from time import time
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.models import SignLanguageCNN, SignLanguageTL, SignLanguageViT
from src.utils import Config

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
        self.model = model.to(device)#.to(torch.float16)
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
              patience:int=None,
              print_every:int=1,
              step_per:Literal['epoch','batch','metric']='epoch',
              step_after:int=5, # Step after this #epoch without improvement when step_per = 'metric'
              save:Literal['best','last']=None
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
        # assert save_best and self.save_path or not save_best, "Save path must be provided if save_best is True."
        assert save and self.save_path or not save, "Save path must be provided if save_best is True."

        train_losses, val_losses = [], []
        best_val_loss = float('inf')  # Track the best validation loss
        # best_acc = 0
        early_stopping_counter = 0
        t = time()
        
        print(f"Training {self.model.__class__.__name__} model using: epochs={epochs} --",
              f"patience={patience} -- print_every={print_every} -- step_per={step_per} -- step_after={step_after}")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = train_criterion(outputs, labels)
                # Backpropagation
                loss.backward()
                # Gradient clipping
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Update weights
                optimizer.step()
                # Update learning rate
                if scheduler is not None and step_per == 'batch':
                    scheduler.step()

                train_loss += loss.item()
            
            train_loss /= len(train)
            val_loss = self.evaluate(val, val_criterion, is_test=False)
            
            # Update learning rate
            if scheduler is not None and step_per == 'epoch':
                scheduler.step(val_loss)
                
            
            # Save losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch == 0 or (epoch+1) % print_every == 0:
                print(f"Epoch [{epoch+1:>3}/{epochs}] Loss: {train_loss:.5f} | Val loss: {val_loss:.5f}",
                      f" | LR: {scheduler.get_last_lr()[0]:.6f}" if scheduler is not None else "",
                      f" -- Best model {'saved' if save=='best' else ''}" if val_loss < best_val_loss
                      else f" -- Best val loss: {best_val_loss:.5f} | {f'Patience: {early_stopping_counter+1}/{patience}'
                    #   f" -- Best model {'saved' if save=='best' else ''}" if acc >= best_acc
                    #   else f" -- Best acc: {best_acc:5.2f} | {f'Patience: {early_stopping_counter+1}/{patience}'
                      if patience is not None else f'Since: {early_stopping_counter+1}'}",
                      sep='')
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            # if acc >= best_acc:
            #     best_acc = acc
                early_stopping_counter = 0
                if save=='best':
                    self.save_model(self.save_path)
            else:
                early_stopping_counter += 1
                if patience is not None and early_stopping_counter >= patience:
                    print(f"Early stopping after epoch {epoch+1}")
                    break
                if scheduler is not None and step_per == 'metric' and early_stopping_counter % step_after == 0:
                    scheduler.step()

        if save == 'last':
            self.save_model(self.save_path)
        elif save == 'best':
            self.load_model()
        
        duration = str(datetime.timedelta(seconds=int(time()-t)))
        print(f'Number of epochs: {epoch} | Total duration: {duration}')
        
        return train_losses, val_losses

    def evaluate(self,
                 val:DataLoader,
                 criterion:nn.modules.loss._Loss=None,
                 is_test:bool=False
            ) -> float | tuple[float, list[int], list[int]]:
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
                    predicted = torch.argmax(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                else:
                    val_loss += criterion(outputs, labels).item()
            
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
                              model_type:str=None,
                              accuracy:float=None,
                              save_dir:str=None,
                              show:bool=False,
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
        ax.margins(0)
        ax.set_adjustable("box")
        cm_df = pd.DataFrame(cm, classes, classes)
        sns.heatmap(cm_df, annot=True, **kwargs)
        plt.title(f"Confusion matrix -- {model_type} model -- acc: {accuracy:.2f}%")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        if show:
            plt.show()
        if save_dir:
            assert model_type and accuracy, "Model type and accuracy must be provided to save the confusion matrix."
            # os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"cm_{model_type}_{accuracy:.2f}.png"), bbox_inches='tight', dpi=300)
    
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
        
    def plot_losses(self,
                    train_losses:list[float],
                    val_losses:list[float],
                    model_type:str=None,
                    save_dir:str=None,
                    show:bool=False) -> None:
        """
        Plot the training and validation losses.
        """
        plt.plot(train_losses, label="train loss")
        plt.plot(val_losses, label="val loss")
        plt.title(f"{model_type} model losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if show:
            plt.show()
        if save_dir:
            assert model_type, "Model type must be provided to save the losses."
            # os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"losses_{model_type}_{int(time())}.png"), bbox_inches='tight', dpi=300)


def get_trainer(model_type:Literal['cnn', 'tl', 'vit'], classes:list[str]) -> SignLanguageModelTrainer:
    num_classes = len(classes)
    if model_type == 'cnn':
        model = SignLanguageCNN(num_classes)
    elif model_type == 'tl':
        model = SignLanguageTL(num_classes, Config.pretrained_model, Config.weights)
    elif model_type == 'vit':
        model = SignLanguageViT(num_classes,
                                d_model         = Config.D_MODEL,
                                ffn_hidden_size = Config.FFN_HIDDEN_SIZE,
                                img_size        = Config.INPUT_SHAPE,
                                patch_size      = Config.PATCH_SIZE,
                                n_channels      = Config.NB_CHANNELS,
                                n_heads         = Config.NB_HEADS,
                                n_layers        = Config.NB_LAYERS,
                                drop_prob       = Config.DROP_PROB,
                                learn_pos_enc   = Config.LEARN_POS_ENC)
    else:
        raise ValueError(f"Model type {model_type} not implemented yet.")
    
    return SignLanguageModelTrainer(model, classes,
                                    save_path=Config.MODEL_SAVE_PATH[model_type],
                                    device=Config.DEVICE)
