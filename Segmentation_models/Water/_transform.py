from torchvision import transforms
import numpy as np
from PIL import Image
import albumentations as A


def get_transforms():
    dim = 320

    clahe_transform = A.CLAHE(p=1)  # CLAHE transformation
    grid_distortion = A.GridDistortion(p=1)  # Grid Distortion transformation
    optical_distortion = A.OpticalDistortion(p=1)  # Optical Distortion transformation
    elastic_transform = A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03) # Elastic transformation
    random_brightness_contrast = A.RandomBrightnessContrast(p=0.5) # Random brightness and contrast

    def apply_custom_transforms(img):
        #img_out = clahe_transform(image=np.array(img))["image"]
        #img_out = grid_distortion(image=img_out)["image"]  # Apply Grid Distortion
        #img_out = optical_distortion(image=img_out)["image"]  # Apply Optical Distortion
        #img_out = elastic_transform(image=img_out)["image"]  # Apply Elastic Transformation
        #img_out = random_brightness_contrast(image=img_out)["image"]  # Apply Random Brightness and Contrast
        return img #Image.fromarray(img_out)

    train_image_transform = transforms.Compose([
        lambda img: apply_custom_transforms(img),
        transforms.Resize((dim, dim)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_mask_transform = transforms.Compose([
        lambda img: apply_custom_transforms(img), # Applying CLAHE to the mask
        transforms.Resize((dim, dim)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    val_image_transform = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_mask_transform = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
    ])

    return train_image_transform, train_mask_transform, val_image_transform, val_mask_transform
