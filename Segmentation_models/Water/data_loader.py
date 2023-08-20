import torch
from torch.utils.data import DataLoader, random_split
from _transform import get_transforms
from dataset import WaterSegmentationDataset

def get_datasets(root_path, val_split=0.2):
    """
    Splits the dataset into training and validation sets.

    :param root_path: Path to the root directory of the dataset
    :param val_split: Fraction of the dataset to be used for validation
    :return: Training and validation datasets
    """
    train_image_transform, train_mask_transform, val_image_transform, val_mask_transform = get_transforms()

    # Create training and validation datasets with appropriate transforms
    train_dataset = WaterSegmentationDataset(root_path, transform=train_image_transform, mask_transform=train_mask_transform)
    val_dataset = WaterSegmentationDataset(root_path, transform=val_image_transform, mask_transform=val_mask_transform)

    val_len = int(val_split * len(train_dataset))
    train_len = len(train_dataset) - val_len
    train_dataset, _ = random_split(train_dataset, [train_len, val_len])
    _, val_dataset = random_split(val_dataset, [val_len, train_len])

    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size=4):
    """
    Creates data loaders for training and validation datasets.

    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param batch_size: Batch size for the data loaders
    :return: Training and validation data loaders
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
