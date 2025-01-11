from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class DataPreprocessor:
    def __init__(self,
                 data_dir:str,
                 transform=None,
                 input_shape:tuple[int, int]=(224, 224),
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 batch_size=32):
        """
        Initialize the DataPreprocessor.

        Args:
            data_dir (str): Path to the dataset directory.
            transform (torchvision.transforms.Compose): Transformations to apply to the images.
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.
            test_ratio (float): Proportion of data to use for testing.
            batch_size (int): Batch size for DataLoader.
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize((0,),(1,))
        ])
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size

    def preprocess(self):
        """
        Process the data into train, validation, and test splits.

        Returns:
            tuple: DataLoader objects for train, validation, and test sets.
        """
        # Load the entire dataset
        full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        
        # Compute dataset sizes
        total_size = len(full_dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # Split the dataset indices
        indices = list(range(total_size))
        train_indices, temp_indices = train_test_split(indices, test_size=(val_size + test_size), random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size, random_state=42)

        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # Create DataLoaders
        train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train, val, test