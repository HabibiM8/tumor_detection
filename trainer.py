import torch
from torch.utils.data import random_split, DataLoader

from main import debug


class Trainer:
    def __init__(self, model, dataset,  criterion = None, optimizer = None, n_epochs: int =10, lr=1e-3, train_percent = 0.7, batch_size: int = 4, debug: bool = False):
        self.model = model
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_percent = train_percent
        self.batch_size = batch_size
        self.debug = debug
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), self.lr)
        self.dataset = dataset
        self.train_size = int(self.train_percent * len(dataset))
        self.test_size = len(dataset) - self.train_size

   #TODO: could hardcode the atrributes into constructor, then use this method as a classmethod, to share attributes across instances
    def prepare_data(self):
        train_set, test_set = random_split(self.dataset, [self.train_size, self.test_size])
        trainLoader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        testLoader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

        if self.debug:

            try:
                # Example to iterate over the training data (3 times only)
                for i, (images, labels, tumor_masks) in zip(range(3), trainLoader):
                    print(f"Iteration {i + 1}:")
                    print("Batch of images shape:", images.shape)  # Shape will be (batch_size, num_channels, height, width)
                    print("Batch of labels:", labels)  # Labels for the batch
                    print("Batch of tumor masks shape:", tumor_masks.shape)  # Shape will be similar to images
                    # You can add your training code here

                # Iterate through test data (3 times only)
                for i, (images, labels, tumor_masks) in zip(range(3), testLoader):
                    print(f"Iteration {i + 1}:")
                    print("Image batch shape:", images.shape)
                    print("Labels batch:", labels)
                    print("Tumor mask batch shape:", tumor_masks.shape)

                # Iterate through train data to print unique labels (3 times only)
                for i, (_, labels, _) in zip(range(3), trainLoader):
                    print(f"Iteration {i + 1}:")
                    print(torch.unique(labels))  # Print values in the range [0, num_classes-1]
                    # Check only the first batch, continue not needed since it runs only 3 times

        return trainLoader, testLoader

