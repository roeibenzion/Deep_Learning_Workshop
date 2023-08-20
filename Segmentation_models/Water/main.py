import argparse
from model import vgg16bn_unet, freeze_encoder
from data_loader import get_datasets, get_dataloaders
from train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model using VGG16BN U-Net architecture")
    parser.add_argument('--root_path', default='/content', help='Root path for the datasets')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--saved_model_name', default='.pth', help='Name of the saved model file')
    parser.add_argument('--steps_per_epoch', type=int, default=550, help='Number of steps per epoch')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer')

    args = parser.parse_args()

    model = vgg16bn_unet(output_dim=1, pretrained=True)
    freeze_encoder(model)

    train_dataset, val_dataset = get_datasets(args.root_path)
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=args.batch_size)

    train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, lr=args.lr, saved_model_name=args.saved_model_name, steps_per_epoch=args.steps_per_epoch, weight_decay=args.weight_decay)

if __name__ == "__main__":
    main()

