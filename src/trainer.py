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
    """
    Trainer class for training and evaluating the different models.
    """
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
            classes (list): List of class names.
            init_model (bool): Whether to initialize the model parameters.
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
        """
        Initialize the model parameters with Xavier initialization.
        """
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
              step_per:Literal['epoch','batch']='epoch',
              step_metric:Literal['loss','accuracy']='loss',
              save:Literal['best','last']=None
    ) -> tuple[list[float], list[float]]:
        """
        Train the model.
        Args:
            train (DataLoader): Training data loader.
            val (DataLoader): Validation data loader.
            train_criterion (_Loss): Training loss function.
            val_criterion (_Loss): Validation loss function.
            optimizer (Optimizer): Optimizer.
            scheduler (LRScheduler): Learning rate scheduler.
            epochs (int): Number of training epochs.
            patience (int): Patience for early stopping.
            print_every (int): Print loss every n epochs.
            step_per (Literal['epoch','batch']): Step learning rate scheduler after per epoch, batch.
            step_metric (Literal['loss','accuracy']): Metric to use for learning rate scheduler if it's a ReduceLROnPlateau.
            save (Literal['best','last']): Save the best or last model.
        Returns:
            tuple[list[float], list[float]]: Training and validation losses.
        """
        assert save and self.save_path or not save, "Save path must be provided if save_best is True."

        train_losses, val_losses = [], []
        best_val_loss = float('inf')  # Track the best validation loss
        best_acc = 0
        early_stopping_counter = 0
        t = time()
        
        print(f"Training {self.model.__class__.__name__} model using: epochs={epochs} --",
              f"patience={patience} -- print_every={print_every} -- step_per={step_per}")
        
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
                    if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                        scheduler.step(val_loss if step_metric == 'loss' else acc)
                    else:
                        scheduler.step()

                train_loss += loss.item()
            
            train_loss /= len(train)
            acc, val_loss = self.evaluate(val, val_criterion, is_test=False)
            
            # Update learning rate
            if scheduler is not None and step_per == 'epoch':
                if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(val_loss if step_metric == 'loss' else acc)
                else:
                    scheduler.step()
                
            
            # Save losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch == 0 or (epoch+1) % print_every == 0:
                print(f"Epoch [{epoch+1:>3}/{epochs}] Loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | Acc: {acc:>5.2f}",
                    f" | LR: {scheduler.get_last_lr()[0]:.6f}" if scheduler is not None else "",
                    f" -- Best model {'saved' if save=='best' else ''}"
                    if (val_loss < best_val_loss if step_metric == 'loss' else acc > best_acc)
                    else f" -- Best {f'val loss: {best_val_loss:.5f}' if step_metric == 'loss' 
                    else f'acc: {best_acc:.5f}'} | {f'Patience: {early_stopping_counter+1}/{patience}'
                    if patience is not None else f'Since: {early_stopping_counter+1}'}",
                    sep='')
            
            if step_metric == 'loss' and val_loss < best_val_loss or step_metric == 'accuracy' and acc > best_acc:
                early_stopping_counter = 0
                if step_metric == 'loss':
                    best_val_loss = val_loss
                else:
                    best_acc = acc
                if save=='best':
                    self.save_model(self.save_path)
            else:
                early_stopping_counter += 1
                if patience is not None and early_stopping_counter >= patience:
                    print(f"Early stopping after epoch {epoch+1}")
                    break
        if save == 'last':
            self.save_model(self.save_path)
        elif save == 'best':
            self.load_model()
        
        duration = str(datetime.timedelta(seconds=int(time()-t)))
        print(f'Number of epochs: {epoch+1} | Total duration: {duration}')
        
        return train_losses, val_losses

    def evaluate(self,
                 val:DataLoader,
                 criterion:nn.modules.loss._Loss=None,
                 is_test:bool=False
            ) -> tuple[float, float] | tuple[float, list[int], list[int]]:
        """
        Evaluate the model.
        Args:
            val (DataLoader): Validation data loader.
            criterion (_Loss): Loss function.
            is_test (bool): Whether to evaluate on test data.
        Returns:
            tuple[float, float] | tuple[float, list[int], list[int]]: Accuracy and loss (if is_test is False)
            or true labels and predicted labels otherwise.
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
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                    
                if is_test:
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                else:
                    val_loss += criterion(outputs, labels).item()
        acc, val_loss = 100 * correct / total, val_loss / len(val)
        return (acc, y_true, y_pred) if is_test else (acc, val_loss)
    
    def predict(self, input:torch.Tensor) -> str:
        """
        Predict the class of an input
        Args:
            input (torch.Tensor): Input image tensor.
        Returns:
            str: Predicted class.
        """
        self.model.to(self.device)
        input = input.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            pred_idx = torch.argmax(output, 1).item()
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
            model_type (str): Model type.
            accuracy (float): Model accuracy.
            save_dir (str): Directory to save the confusion matrix plot.
            show (bool): Whether to display the plot.
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
        Load the model from a file.
        Args:
            path (str): Path to the model file.
        """
        assert path or self.save_path, "Save path must be provided to load the model."
        
        path = path or self.save_path
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
    def save_model(self, path:str=None) -> None:
        """
        Save the model to a file.
        Args:
            path (str): Path to save the model.
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
        Args:
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses.
            model_type (str): Model type.
            save_dir (str): Directory to save the plot.
            show (bool): Whether to display the plot.
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


def get_trainer(model_type:Literal['cnn','tl','vit'], classes:list[str]) -> SignLanguageModelTrainer:
    """
    Get the model trainer based on the model type.
    Args:
        model_type (Literal['cnn','tl','vit']): Model type.
        classes (list): List of class names.
    Returns:
        SignLanguageModelTrainer: Model trainer.
    """
    num_classes = len(classes)
    if model_type == 'cnn':
        model = SignLanguageCNN(num_classes, Config.NB_CHANNELS)
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
