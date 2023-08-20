import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import shutil
import torch
import torch.nn as nn
#!pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117

from torch.utils.data import random_split
from numba import cuda