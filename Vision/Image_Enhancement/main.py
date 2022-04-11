import numpy as np
from numpy.core.numeric import full
import pandas as pd

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

import gc
import os
import math
import random
from glob import glob
from tqdm.notebook import tqdm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.loss import _Loss

import segmentation_models_pytorch as smp

from torchvision.models import vgg16
from typing import Tuple, List

from utils import seed_everything
from data import train_valid_split

def main(CFG):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('[Current Divice] :',device)

    seed_everything(CFG.seed)
    print('SEED :', CFG.seed)

    train_input_paths, train_label_paths, valid_input_paths, valid_label_paths = train_valid_split(valid_type=1, full_train=False)
