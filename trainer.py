import torch
from torch.utils.data import random_split, DataLoader



class Trainer:
    def __init__(self, model, dataset, device, criterion = None, optimizer = None, n_epochs: int =10, lr=1e-3, train_percent = 0.7, batch_size: int = 4, debug: bool = False):
        self.model = model
        self.device = device
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

    def prepare_data(self):
        self.train_set, self.test_set = random_split(self.dataset, [self.train_size, self.test_size])
        self.trainLoader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.testLoader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

        if self.debug:

            try:
                for i, (images, labels, tumor_masks) in zip(range(3), self.trainLoader):
                    print(f"Iteration {i + 1}:")
                    print("Batch of images shape:", images.shape)  # Shape (batch_size, num_channels, height, width)
                    print("Batch of labels:", labels)
                    print("Batch of tumor masks shape:", tumor_masks.shape)

                for i, (images, labels, tumor_masks) in zip(range(3), self.testLoader):
                    print(f"Iteration {i + 1}:")
                    print("Image batch shape:", images.shape)
                    print("Labels batch:", labels)
                    print("Tumor mask batch shape:", tumor_masks.shape)

                for i, (_, labels, _) in zip(range(3), self.trainLoader):
                    print(f"Iteration {i + 1}:")
                    print(torch.unique(labels))

            except Exception as e:
                print(f"Error in data Preperation, check shapes and sizes!: {e}")

            finally:
                print("Debugging complete")

        return self.trainLoader, self.testLoader


    def loop_training(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels, _ in self.trainLoader:
                images, labels = images.to(self.device), labels.to(self.device).long() - 1  # bc matlab counts from 1

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.n_epochs}], Loss: {running_loss / len(self.trainLoader)}")

        # Save the trained model
        torch.save(self.model.state_dict(), 'resnet_finetuned.pth')



    def loop_testing(self):
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation for faster computation
            for images, labels, _ in self.testLoader:
                images, labels = images.to(self.device), labels.to(self.device).long() - 1  # bc matlab counts from 1
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')