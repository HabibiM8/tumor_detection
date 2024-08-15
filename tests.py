import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import *
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import torch.nn as nn
#path = '~/Uni/SS24/DLAM_Data/Data/data/1'
#MatDataset.explore_mat_file(path)
mat_dir = '~/Uni/SS24/DLAM_Data/Data/data'

dataset = MatDataset(mat_dir)
image, label, tumor_mask = dataset[0]

print(image.shape, label, tumor_mask.shape)

mat_dir = '~/Uni/SS24/DLAM_Data/Data/data'
MatDataset.display_image_from_mat(mat_dir)
