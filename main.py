import torch, torchvision

import torchvision.transforms as transforms

from dataloader import *

path = '~/Uni/SS24/DLAM_Data/Data/data/1'
#MatDataset.explore_mat_file(path)

dataset = MatDataset(path)
image, label, tumor_mask = dataset[0]

print(image.shape, label, tumor_mask.shape)

path = '~/Uni/SS24/DLAM_Data/Data/data'
MatDataset.display_image_from_mat(path)