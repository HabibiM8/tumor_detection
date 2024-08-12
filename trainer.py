
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from DataLoader import CustomDataloader
from model import FullyConnected_AE
from tqdm import tqdm
import sys
class Trainer:
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr =0.001)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data,target) in tqdm(enumerate(self.train_data)):
            self.optimizer.zero_grad()
            if data.shape[0] < 64:  #batch_size == 64 hardcoded, skip last batch if smaller
                continue
            self.output = self.model(data)
            loss = self.criterion(self.output, data.view(data.size(0),-1)) #compare output with original input into the AE
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_data.dataset)} ({100. * batch_idx / len(self.train_data):.0f}%)]\tLoss: {loss.item():.6f}')

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_data:
                if data.shape[0] < 64:  # batch_size == 64 hardcoded, neglect the last batch if it is smaller than the other batches
                    continue
                output = self.model(data)
                test_loss += self.criterion(output, data.view(data.size(0),-1)).item()
                pred = output.argmax(dim=1, keepdim = True)
                correct += pred.argmax().sum().item()
        test_loss /= len(self.test_data.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_data.dataset)} ({100. * correct / len(self.test_data.dataset):.0f}%)\n')

    def loop(self):
        for epoch in range(1,5):
            self.train(epoch)
            #self.test()