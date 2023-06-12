import numpy as np
import os 
import cv2
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data


class data_set(torch.utils.data.Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.root_dir, self.images[idx]))
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # zero the parameter gradients
        output = model(data) # forward pass
        loss = criterion(output, target) # calculate loss
        loss.backward() # backward pass
        optimizer.step() # optimize weights
    
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # we don't need to calculate gradients for the test set
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # forward pass
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # compare predictions to true label
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def prepare_data(relevant_data_pth, non_relevant_data_pth):
    relevant_data = os.listdir(relevant_data_pth)
    non_relevant_data = os.listdir(non_relevant_data_pth)
    relevant_data = [os.path.join(relevant_data_pth, image) for image in relevant_data]
    non_relevant_data = [os.path.join(non_relevant_data_pth, image) for image in non_relevant_data]
    #return as a labled dataset
    concat = []
    for image in relevant_data:
        concat.append((image, 1))
    for image in non_relevant_data:
        concat.append((image, 0))
    return concat

def main():
    #hyperparameters
    lr = 0.001
    epochs = 10
    batch_size = 32

    #define transforms
    transform =   transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
    #prepare data
    concat = prepare_data("data/relevant", "data/non_relevant")
    #split data to train and test
    train_size = 0.8*(len(concat))
    test_size = len(concat) - train_size
    train_set, test_set = torch.utils.data.random_split(concat, [train_size, test_size])
    train_set = data_set(ds=train_set, transform=transform)
    test_set = data_set(ds=test_set, transform=transform)
    #define data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    #define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #define model using transfer learning
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier
    model = model.to(device)
    #define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    #train and test
    train(model, train_loader, optimizer, criterion, device)
    test_loss, accuracy = test(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, accuracy: {accuracy:.4f}%")
    #save model if it was better the 0.8 accuracy
    if accuracy > 80:
        torch.save(model.state_dict(), "model.pkl")
        print("model saved")
    
