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

#dataset = MatDataset(path)
#image, label, tumor_mask = dataset[0]

#print(image.shape, label, tumor_mask.shape)


"""please view https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/data"""

mat_dir = '~/Uni/SS24/DLAM_Data/Data/data'
#MatDataset.display_image_from_mat(path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


#########

for _, labels, _ in trainLoader:
    print(torch.unique(labels))  # This should print values in the range [0, num_classes-1]
    break  # Check only the first batch

########






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




##model

resnet = models.resnet18(pretrained=True)  #TODO use ResNet 50 later

num_classes = 3 #with or without tumor


#change final fully conected layer for num of classes
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
"""
for param in resnet.parameters():
    param.requires_grad = True"""


# Freeze all layers except the final fully connected layer
for param in resnet.parameters():
    param.requires_grad = False

# Only the final layer will have requires_grad = True
for param in resnet.fc.parameters():
    param.requires_grad = True


criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)


num_epochs = 10  # Number of epochs to train

resnet = resnet.to(device)  # Move model to GPU if available




#training loop
for epoch in range(num_epochs):
    resnet.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels, _ in trainLoader:
        images, labels = images.to(device), labels.to(device).long() - 1 #bc matlab counts from 1

        optimizer.zero_grad()

        outputs = resnet(images)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainLoader)}")

# Save the trained model
torch.save(resnet.state_dict(), 'resnet_finetuned.pth')




#eval loop

resnet.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for faster computation
    for images, labels, _ in testLoader:
        images, labels = images.to(device), labels.to(device).long() - 1 #bc matlab counts from 1
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
