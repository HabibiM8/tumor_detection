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


#TODO: tensorboard, argparser and config

"""please view https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset/data"""

debug = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mat_dir = '~/Uni/SS24/DLAM_Data/Data/data'
MatDataset.display_image_from_mat(mat_dir)



#######loader

transform = Transform.get_transform()

dataset = MatDataset(mat_dir, transform=transform)


model = ResNetModel(device= device, model_size=18, fine_tune_entire_network=False).get_model()
trainer = Trainer(model = model,
                  dataset= dataset,
                  device= device,
                  criterion = torch.nn.CrossEntropyLoss,
                  n_epochs = 10,
                  lr = 1e-3,
                  train_percent = 0.7,
                  batch_size = 4,
                  debug = debug
                  )
trainLoader, testLoader = trainer.prepare_data()

trainer.loop_training()
trainer.loop_testing()



