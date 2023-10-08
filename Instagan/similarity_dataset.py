import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copyfile, copytree

# Set the device (CPU or GPU) for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained ResNet-18 model
model = resnet18(pretrained=True)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load images from trainA and trainB
trainA_path = 'path_to_directory/trainA'
trainB_path = 'path_to_directory/trainB'

# Sample reference images from trainB
k = 1500  # Number of reference images to sample
reference_images = [os.path.join(trainB_path, f) for f in os.listdir(trainB_path) if f.endswith('.jpg')][:k]

# Extract features for trainA and reference images
trainA_features = []
reference_features = []

# Define trainA_images by listing the files in trainA
trainA_images = [os.path.join(trainA_path, f) for f in os.listdir(trainA_path) if f.endswith('.jpg')]

for img_path in trainA_images:
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    trainA_features.append(features.cpu().numpy())

for img_path in reference_images:
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    reference_features.append(features.cpu().numpy())

# Calculate the average feature map for trainB images
average_feature_map_B = np.mean(reference_features, axis=0)

# Calculate cosine similarity between trainA features and the average feature map of trainB
similarities = cosine_similarity(np.array(trainA_features).reshape(len(trainA_features), -1),
                                 average_feature_map_B.reshape(1, -1))

# Flatten the similarities array to get a 1D array
similarities = similarities.flatten()

# Sort images by similarity and select the top-k similar images from trainA
top_similar_indices = np.argsort(similarities)[::-1][:k]
top_similar_images = [(i, similarities[i]) for i in top_similar_indices]

# Create a 'clean' directory and copy the selected images and masks
clean_dir = 'path_to_directory/clean'
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(os.path.join(clean_dir, 'trainA'), exist_ok=True)
os.makedirs(os.path.join(clean_dir, 'trainA_seg'), exist_ok=True)
os.makedirs(os.path.join(clean_dir, 'trainB'), exist_ok=True)
os.makedirs(os.path.join(clean_dir, 'trainB_seg'), exist_ok=True)

# Copy selected trainA images and corresponding masks to 'clean/trainA' and 'clean/trainA_seg'
for i, _ in top_similar_images:
    img_filename = os.path.basename(trainA_images[i])
    mask_filename = img_filename.replace('.jpg', '.png')
    copyfile(trainA_images[i], os.path.join(clean_dir, 'trainA', img_filename))
    mask_src_path = os.path.join('path_to_directory/trainA_seg', mask_filename)
    mask_dest_path = os.path.join(clean_dir, 'trainA_seg', mask_filename)
    copyfile(mask_src_path, mask_dest_path)

# Copy all trainB images and masks to 'clean/trainB' and 'clean/trainB_seg'
copytree(trainB_path, os.path.join(clean_dir, 'trainB'))
copytree('path_to_directory/trainB_seg', os.path.join(clean_dir, 'trainB_seg'))
