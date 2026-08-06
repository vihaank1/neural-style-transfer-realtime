import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_SIZE = 128

train_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),  # [0,1]
])

class ImageFolderFlat(Dataset):
    """Loads all images from a flat list of file paths."""
    def __init__(self, file_list, transform=train_transform):
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

def load_file_list(list_path):
    with open(list_path) as f:
        return [line.strip() for line in f if line.strip()]
