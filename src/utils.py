import math

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset as HFDataset

# Config
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    SEED = 42
    DATA_PATH = "Hemg/sign_language_dataset"
    MODEL_SAVE_PATH = {
        'cnn': 'assets/models/sign_language_cnn.pth',
        'tl' : 'assets/models/sign_language_tl.pth',
        'vit': 'assets/models/sign_language_vit.pth',
    }
    OUTPUT_DIR = "assets/outputs"
    
    pretrained_model = models.resnet18
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    
    VAL_RATIO, TEST_RATIO = 0.15, 0.15
    EPOCHS = 100
    PATIENCE = 20
    PRINT_EVERY = 5
    STEP_PER = {'cnn': 'epoch', 'tl': 'epoch', 'vit': 'epoch'}
    STEP_AFTER = 5
    STEP_METRIC = 'loss'
    STEP_MODE = {'loss': 'min', 'accuracy': 'max'}
    STEP_FACTOR = 0.8
    WARMUP_STEPS = 1000
    NB_WORKERS = 0 # Bad for value > 0 on MacOS
    LR = {'cnn': 0.001, 'tl': 0.001, 'vit': 0.0001}
    LABEL_SMOOTHING = {'cnn': 0.025, 'tl': 0.025, 'vit': 0.025}
    BETAS = (0.9, 0.98)
    EPS = 1e-9
    SAVE_MODEL = 'best' # 'best' or 'last'
    
    INPUT_SHAPE = (224, 224)
    NB_CHANNELS = 3
    BATCH_SIZE = 32
    PATCH_SIZE = 16
    
    D_MODEL = 320
    NB_HEADS = D_MODEL // 64
    FFN_HIDDEN_SIZE = D_MODEL * 4
    NB_LAYERS = 6
    DROP_PROB = 0.1
    LEARN_POS_ENC = True
    
    FIG_SIZE = (14, 12)



# Attention
def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]
    scaled = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention_weights = F.softmax(scaled, dim=-1)
    values = attention_weights @ v
    return values, attention_weights



# Performance
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
        lr = self.calc_lr(self._step_count)
        return [lr] * self.num_param_groups

    def calc_lr(self, step):
        return self.dim_embed**(-0.5) * min(step**(-0.5), step * self.warmup_steps**(-1.5))



# Data
class SignLanguageDataset(Dataset):
    def __init__(self, dataset: HFDataset) -> None:
        assert isinstance(dataset, HFDataset), "Dataset must be a Hugging Face Dataset object"
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.dataset[idx]
        return (item["image"], item["label"])

class DataPreprocessor:
    def __init__(self,
                 path:str,
                 transform:transforms.Compose=None,
                 resize_shape:tuple[int,int]=(224, 224)):
        """
        Initialize the DataPreprocessor.

        Args:
            path (str): Path to the huggingface dataset.
            transform (torchvision.transforms.Compose): Transformations to apply to the images.
            resize_shape (tuple[int,int]): Shape to resize the input images to.
        """
        
        self.dataset = load_dataset(path, split="train")
        self.classes = self.dataset.features["label"].names
        
        self.transform = transform or transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset.set_transform(self.apply_transform, columns=["image"], output_all_columns=True)

    def preprocess(self,
                   dataset:HFDataset=None,
                   test_ratio:float=0.2,
                   val_ratio:float=0.0,
                   batch_size:int=32,
                   seed:int=42,
                   **kwargs
            ) -> tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]:
        """
        Process the data into train, validation, and test splits.
        
        Args:
            dataset (HFDataset): The dataset to process.
            test_ratio (float): The ratio of the test set.
            val_ratio (float): The ratio of the validation set.
            batch_size (int): The batch size.
            seed (int): The random seed.
            **kwargs: Additional arguments to pass to the DataLoader.

        Returns:
            tuple: DataLoader objects for train, validation, and test sets.
        """

        # Split the dataset into train, val, and test sets
        dataset = self.split_dataset(dataset or self.dataset, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed)

        train = self.dataloader(dataset["train"], batch_size=batch_size, shuffle=True, **kwargs)
        val = self.dataloader(dataset["val"], batch_size=batch_size, **kwargs) if val_ratio > 0 else None
        test = self.dataloader(dataset["test"], batch_size=batch_size, **kwargs)

        return (train, val, test) if val_ratio > 0 else (train, test)
    
    def dataloader(self, dataset:HFDataset, batch_size:int=32, **kwargs) -> DataLoader:
        """ Create a DataLoader for a Hugging Face Dataset object\n
        Args:
            dataset (HFDataset): The dataset to create a DataLoader for
            batch_size (int): The batch size
            **kwargs: Additional arguments to pass to the DataLoader
        Returns:
            DataLoader: A DataLoader for the dataset"""
        return DataLoader(SignLanguageDataset(dataset), batch_size=batch_size, num_workers=Config.NB_WORKERS, **kwargs)
    
    def split_dataset(self,
                      dataset:HFDataset,
                      test_ratio:float=0.2,
                      val_ratio:float=0.0,
                      seed:int=42
            ) -> DatasetDict:
        
        assert isinstance(dataset, HFDataset), "Dataset must be a Hugging Face DatasetDict object"
        
        train_test = dataset.train_test_split(test_size=test_ratio, stratify_by_column="label", seed=seed)
        train_val  = train_test["train"].train_test_split(test_size=val_ratio, stratify_by_column="label", seed=seed) if val_ratio > 0 else None

        # Combine splits into a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_val["train"],
            "val": train_val["test"],
            "test": train_test["test"]} if val_ratio > 0 else {
            "train": train_test["train"],
            "test": train_test["test"]
        })
        
        return dataset_dict
    
    def apply_transform(self, batch:dict) -> dict:
        """ Apply the transformation to a batch of images\n
        Args:
            batch (torch.Tensor): The batch of images to transform
        Returns:
            torch.Tensor: The transformed batch"""
            
        return {'image': [self.transform(item) for item in batch['image']]}
    
    def clear_cache(self) -> None:
        """ Clear the cache of all Hugging Face datasets"""
        self.dataset.cleanup_cache_files()