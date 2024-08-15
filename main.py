import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import *
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import torch.nn as nn

from model import ResNetModel
from trainer import Trainer

#path = '~/Uni/SS24/DLAM_Data/Data/data/1'
#MatDataset.explore_mat_file(path)

#dataset = MatDataset(path)
#image, label, tumor_mask = dataset[0]

#print(image.shape, label, tumor_mask.shape)


"""please view https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/data"""

debug = False

mat_dir = '~/Uni/SS24/DLAM_Data/Data/data'
#MatDataset.display_image_from_mat(path)



#######loader

transform = Transform.get_transform()

dataset = MatDataset(mat_dir, transform=transform)


model = ResNetModel(model_size=18, fine_tune_entire_network=False).get_model()
trainer = Trainer(model = model,
                  dataset= dataset,
                  criterion = torch.nn.CrossEntropyLoss,
                  n_epochs = 10,
                  lr = 1e-3,
                  train_percent = 0.7,
                  batch_size = 4,
                  debug = debug
                  )
trainLoader, testLoader = trainer.prepare_data()



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
