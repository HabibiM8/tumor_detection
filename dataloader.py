import os
from functools import lru_cache

import torch
from tensorboard.data.server_ingester import DataServerStartupError
from torch.utils.data import Dataset
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

class MatDataset(Dataset):
    def __init__(self, mat_dir, transform = None):
        self.mat_dir = os.path.expanduser(mat_dir)
        self.mat_files = [f for f in os.listdir(self.mat_dir) if f.endswith('.mat')]
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        #with h5py.File(self.mat_file, 'r') as mat_file:
        return len(self.mat_files)

    @lru_cache(maxsize=300)
    def __getitem__(self, index):
        mat_file_path = os.path.join(self.mat_dir, self.mat_files[index])
        with h5py.File(mat_file_path, 'r') as mat_file:
            image = mat_file['cjdata/image'][()]
            label = mat_file['cjdata/label'][()] #.astype(int)
            tumor_mask = mat_file['cjdata/tumorMask'][()]

            image = Image.fromarray(image)
            tumor_mask = Image.fromarray(tumor_mask)

            if image.size != (512,512):
                resize_transform = transforms.Resize((512,512))
                image = resize_transform(image)
                tumor_mask = resize_transform(tumor_mask)


            if self.transform:
                image = self.transform(image)
                tumor_mask = self.transform(tumor_mask)
            return image, label, tumor_mask

    @classmethod
    def get_mat_key(self,path): # = os.getcwd()):
        expanded_path = os.path.expanduser(path) + '.mat'
        with h5py.File(expanded_path, 'r') as mat_file:
            data = mat_file['1'][()]


    @classmethod
    def explore_mat_file(cls, path):
        expanded_path = os.path.expanduser(path) + '.mat'
        with h5py.File(expanded_path, 'r') as mat_file:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")

            mat_file.visititems(print_structure)


    @classmethod
    def display_image_from_mat(cls, mat_file_path):

        mat_file_path = os.path.expanduser(mat_file_path)
        mat_file_path = os.path.join(mat_file_path, '1.mat')
        with h5py.File(mat_file_path, 'r') as mat_file:
            image_data = mat_file['cjdata/image'][()]

            image = Image.fromarray(image_data)

            plt.imshow(image, cmap='gray')
            plt.title("Image from .mat file")
            plt.axis('off')
            plt.show()

class Transform:
    def __init__(self):
        pass

    @staticmethod
    def get_transform():
        return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # If grayscale, convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])