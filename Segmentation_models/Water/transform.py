from torchvision import transforms
from scipy import ndimage
import numpy as np
from PIL import Image
import albumentations as A
import cv2

def get_transforms():
    dim = 320

    clahe_transform = A.CLAHE(p=1)
    grid_distortion = A.GridDistortion(p=1)
    optical_distortion = A.OpticalDistortion(p=1)
    elastic_transform = A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    random_brightness_contrast = A.RandomBrightnessContrast(p=0.5)
    gaussian_blur = A.GaussianBlur(p=0.5)
    random_crop = A.RandomCrop(height=dim, width=dim, p=0.5)
    random_resize = A.RandomScale(scale_limit=0.2, p=0.5)

    def apply_custom_transforms(img, mask=None):
        augmented = A.Compose([
            clahe_transform,
            grid_distortion,
            optical_distortion,
            elastic_transform,
            random_brightness_contrast,
            gaussian_blur,
            random_crop,
            random_resize
        ])(image=np.array(img), mask=np.array(mask))

        return Image.fromarray(augmented["image"]), Image.fromarray(augmented["mask"])

    train_transforms = transforms.Compose([
        lambda img, mask: apply_custom_transforms(img, mask),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
        transforms.Resize((dim, dim)),
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

    return train_transforms, mask_transforms, val_image_transform, val_mask_transform
