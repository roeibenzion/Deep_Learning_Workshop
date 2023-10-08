import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

IMSIZE = 512
STYLE_WEIGHT = 1000000
CONTENT_WEIGHT = 1
SHARPNESS_WEIGHT = 0.1
NUM_STEPS = 300


def image_loader(image_name):
    """This function loads an image, resizes it to the desired image size and transforms it to tensor"""
    loader = transforms.Compose([
        transforms.Resize((IMSIZE, IMSIZE)),
        transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(input):
    """This function computes the gram matrix by calculating a dot product between the feature matrix and it's
    transpose and then normalize it by dividing by the number of elements
    it used for style loss calculation"""
    batch, channel, height, width = input.size()
    features = input.view(batch * channel, height * width)
    G = torch.mm(features, features.t())
    return G.div(batch * channel * height * width)


class ContentLoss(nn.Module):
    """ This class calculates the content loss between the content feature representation of the generated image and
    a target content representation.
    """

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """ This class calculates the style loss between the style feature representation of the generated image and
        a target style representation by using the gram matrix of the features.
        """
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """This class represents image normalization (very first layer). It subtracts the mean values and divides by the
    standard deviation values for each color channel. This normalization step helps ensure that the input image has a
    similar distribution of pixel values as the data that the model was originally trained on"""
    def __init__(self):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class SharpnessLoss(nn.Module):
    """This class designed to compute a sharpness loss by measuring the difference between the original input tensor
    and a blurred version of it obtained through average pooling. This loss can be used as a regularization term to
    encourage the generated image to be less blurry and maintain sharpness during style transfer."""

    def forward(self, x):
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        sharpness = torch.mean(torch.abs(x - blurred))
        return sharpness


def get_style_model_and_losses(cnn, style_img, content_img):
    """This function sets up the neural network model with the appropriate normalization, content and style loss
    layers inserted at the desired layers within the model"""
    normalization = Normalization().to(device)
    # Desired layers to calculate content and style loss
    content_layers = ['conv_4', 'conv_6']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # If the current layer is in desired content layers,
        # add it to the model
        if name in content_layers:
            # Calculate the content target and create a content loss module
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # If the current layer is in desired style layers,
        # add it to the sequential model
        if name in style_layers:
            # Calculate the style target and create a style loss module
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim the layers after last content or style layer
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    """This function initializes an optimizer, an LBFGS optimizer, and sets it up to optimize the pixel values of the
    input image tensor"""
    optimizer = optim.LBFGS([input_img])
    return optimizer


def main(cnn, content_img, style_img, input_img, num_steps, style_weight, content_weight, sharpness_weight):
    """This function performs the optimization process to generate an image that combines the content image
    with the style of style image"""
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    # Optimize the input image not the network
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    print('#####Optimizing Image#####')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            sharpness_loss = sharpness_weight * SharpnessLoss()(input_img)

            loss = style_score + content_score + sharpness_loss
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("Epoch {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Sharpness Loss: {:.4f}'.format(
                    style_score.item(), content_score.item(), sharpness_loss.item()))
                print()

            return style_score + content_score + sharpness_loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help="content image", required=True)
    parser.add_argument('-s', help="style image", required=True)
    parser.add_argument('-style_weight', help="style weight", type=int, default=STYLE_WEIGHT)
    parser.add_argument('-content_weight', help="content weight", type=int, default=CONTENT_WEIGHT)
    parser.add_argument('-sharpness_weight', help="sharpness weight", type=float, default=SHARPNESS_WEIGHT)
    parser.add_argument('-steps', help="number of steps", type=int, default=NUM_STEPS)
    parser.add_argument('-save', help="generated image name", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_img = image_loader(args.s)
    content_img = image_loader(args.c)

    # Pretrained CNN model
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    input_img = content_img.clone()
    input_img.requires_grad_(True)
    input_img.data.clamp_(0, 1)

    output = main(cnn, content_img, style_img, input_img, args.steps, args.style_weight, args.content_weight,
                  args.sharpness_weight)

    torchvision.utils.save_image(output, args.save + ".png")

    image = Image.open(args.save + ".png")

    # Define the desired output size (larger size) and save the image with the new size as "output_image.jpg"
    desired_width = 800
    desired_height = 600

    resized_image = image.resize((desired_width, desired_height), Image.ANTIALIAS)
    resized_image.save('output_image.jpg')
    print("Image Saved")
