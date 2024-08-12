import scipy.io
import pandas as pd
import os
import torch

from torchvision import datasets, transforms
"""I recycled a previously implemented DataLoader, so there is still a test data set passed as an agument, but it shall not be used further..."""


class CustomDataloader:
    def __init__(self, train_batch_size, test_batch_size):

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


        self.train_dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.MNIST(root='./data', train=False, transform= self.transform)


        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size= self.train_batch_size, shuffle= True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size= self.test_batch_size, shuffle=False)



    @classmethod
    def getData(cls,train_batch_size, test_batch_size):
        instance = cls(train_batch_size, test_batch_size)
        return instance.train_loader, instance.test_loader



    def mat_to_parquet(mat_file_path, parquet_file_path):
        """load .mat and save to parquet file"""
        for file_name in os.listdir(mat_file_path):
            if file_name.endswith(".mat"):
                mat_file_path = os.path.join(mat_file_path, file_name)

                mat_data = scipy.io.loadmat(mat_file_path)

                for key in mat_data:
                    if isinstance(mat_data[key], (list, dict, pd.DataFrame)):
                        df = pd.DataFrame(mat_data[key])
                        break
                parquet_file_path = os.path.join(parquet_file_path, file_name.replace('.mat', '.parquet'))
                df.to_parquet(parquet_file_path)




