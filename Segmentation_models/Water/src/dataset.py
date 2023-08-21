from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class BaseSegmentationDataset(Dataset):
    def __init__(self, root, image_path, mask_path, transform=None, mask_transform=None):
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_path = image_path
        self.mask_path = mask_path
        self.data = sorted(os.listdir(os.path.join(root, image_path)))
        self.mask = sorted(os.listdir(os.path.join(root, mask_path)))
        if len(self.data) != len(self.mask):
            raise ValueError("Mismatch in the number of images and masks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_path, self.data[idx])
        mask_path = os.path.join(self.root, self.mask_path, self.mask[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        seed = torch.randint(0, 2**32, (1,)).item()

        if self.transform:
            torch.manual_seed(seed)
            img = self.transform(img)

        if self.mask_transform:
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float() # Apply threshold after converting to tensor

        return img, mask

class BeachDataset(BaseSegmentationDataset):
    def __init__(self, root, transform=None, mask_transform=None):
        super().__init__(root, image_path='images', mask_path='masks', transform=transform, mask_transform=mask_transform)

class WaterSegmentationDataset(BaseSegmentationDataset):
    def __init__(self, root, transform=None, mask_transform=None, image_path='images', mask_path='masks'):
        super().__init__(root, image_path=image_path, mask_path=mask_path, transform=transform, mask_transform=mask_transform)
