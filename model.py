import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import *
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import torch.nn as nn


class ResNetModel():

    def __init__(self, model_size: int  = 18, fine_tune_entire_network: bool = False):

        self.fine_tune_entire_network = fine_tune_entire_network

        if model_size == 18:
            self.resnet = models.resnet18(pretrained=True)

        if model_size == 34:
            self.resnet = models.resnet34(pretrained=True)

        if model_size == 50:
            self.resnet = models.resnet50(pretrained=True)

        if model_size == 101:
            self.resnet = models.resnet101(pretrained=True)

        if model_size == 152:
            self.resnet = models.resnet152(pretrained=True)

        self.num_classes = 3 #cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor

        #change final fully conected layer for num of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

        """You can choose to fine-tune the entire network or just train the final layer. 
        Fine-tuning means updating the weights of all the layers, while transfer learning 
        typically means freezing the early layers and only training the final layer"""

        #Fine-tune all layers
        if self.fine_tune_entire_network:
            for param in self.resnet.parameters():
                param.requires_grad = True

        #Freeze all layers except the final fully connected layer
        else:
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.resnet.fc.parameters(): #fully conected
                param.requires_grad = True

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = self.resnet.to(device)


    def get_model(self):
      return self.resnet

