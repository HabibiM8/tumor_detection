import torch, torchvision

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloader import *
import torch
from torch.utils.data import random_split, DataLoader
#path = '~/Uni/SS24/DLAM_Data/Data/data/1'
#MatDataset.explore_mat_file(path)

#dataset = MatDataset(path)
#image, label, tumor_mask = dataset[0]

#print(image.shape, label, tumor_mask.shape)

mat_dir = '~/Uni/SS24/DLAM_Data/Data/data'
#MatDataset.display_image_from_mat(path)



#######loader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # If the image is grayscale, convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = MatDataset(mat_dir, transform=transform)


train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_set, test_set = random_split(dataset, [train_size, test_size])


trainLoader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0)

a = 1
if a == 2:
    # Example to iterate over the training data
    for images, labels, tumor_masks in trainLoader:
        print("Batch of images shape:", images.shape)  # Shape will be (batch_size, num_channels, height, width)
        print("Batch of labels:", labels)  # Labels for the batch
        print("Batch of tumor masks shape:", tumor_masks.shape)  # Shape will be similar to images
        # You can add your training code here

    # Iterate through test data
    for images, labels, tumor_masks in testLoader:
        # Accessing data in the batch
        print("Image batch shape:", images.shape)
        print("Labels batch:", labels)
        print("Tumor mask batch shape:", tumor_masks.shape)