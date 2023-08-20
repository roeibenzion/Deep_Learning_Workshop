from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import autocast
import torch
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
from loss import CombinedLoss1
from tqdm import tqdm

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean
# Define a function to calculate accuracy
def accuracy_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct / target.size(0)

def save_predictions_to_folder(images, masks, outputs, folder_path, threshold=0.5):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Denormalize the images
    images = denormalize(images).cpu()
    masks = masks.cpu().unsqueeze(1)
    outputs = torch.sigmoid(outputs).cpu().detach().unsqueeze(1)

    # Iterate through the images and save them
    for i in range(images.shape[0]):
        image_path = os.path.join(folder_path, f'image_{i}.png')
        mask_path = os.path.join(folder_path, f'mask_{i}.png')
        prediction_path = os.path.join(folder_path, f'prediction_{i}.png')
        prediction_threshold_path = os.path.join(folder_path, f'prediction_threshold_{i}.png')

        save_image(images[i], image_path)
        save_image(masks[i], mask_path)
        save_image(outputs[i], prediction_path)
        save_image((outputs[i] > threshold).float(), prediction_threshold_path)

def train_epoch(model, loader, criterion, optimizer, device, steps_per_epoch=200):
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    # Create a progress bar using tqdm
    progress_bar = tqdm(range(steps_per_epoch), desc="Training", leave=False)

    # Iterate only through the defined number of steps
    for step in progress_bar:
        images, masks = next(iter(loader))
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast(): # Adding autocast for mixed precision
            outputs = model(images)
            loss = criterion(outputs, masks, model)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_accuracy += accuracy_score(outputs, masks)
        # Update the progress bar with the current loss
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return train_loss / steps_per_epoch, train_accuracy / steps_per_epoch

def iou_score(output, target):
    smooth = 1e-6
    if torch.is_tensor(output):
        output = (output > 0.5).int()
    if torch.is_tensor(target):
        target = (target > 0.5).int()
    intersection = (output & target).float().sum((1, 2, 3))
    union = (output | target).float().sum((1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with autocast(): # Adding autocast for mixed precision
                outputs = model(images)
                loss = criterion(outputs, masks, model)
            val_loss += loss.item()
            val_accuracy += accuracy_score(outputs, masks)
            total_iou += iou_score(outputs, masks)
    return val_loss / len(loader), val_accuracy / len(loader), total_iou / len(loader)


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4,display_freq=1, saved_model_name='fine_tuned_unet', patience=10, weight_decay=1e-4, steps_per_epoch=200, save_folder='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = CombinedLoss1()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device, steps_per_epoch)
        val_loss, val_accuracy, val_iou = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()  # Update the learning rate
        save_dir = os.path.join(save_folder, f'{saved_model_name}/epoch_{epoch+1}')
        # Save the best performing model
        if val_loss < best_val_loss:
            print(f"Validation loss decreased from {best_val_loss} to {val_loss}. Saving best model...")
            best_val_loss = val_loss
            model_filename = f'{save_dir}/{saved_model_name}_val_loss{val_loss:.4f}.pth'
            torch.save(model.state_dict(), model_filename)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping if no improvement in validation loss
        if epochs_no_improve == patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            model_filename = f'{save_dir}/{saved_model_name}.pth'
            torch.save(model.state_dict(), model_filename)
            return best_val_loss

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Val IoU: {val_iou}")

        if (epoch + 1) % display_freq == 0:
            images, masks = next(iter(val_loader))
            outputs = model(images.to(device))
            save_predictions_to_folder(images, masks, outputs, f'{save_dir}/{saved_model_name}/epoch_{epoch+1}')
            del images, masks, outputs
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return best_val_loss
