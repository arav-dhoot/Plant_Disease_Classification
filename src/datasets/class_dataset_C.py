import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as T
import numpy as np

class PlantImageDatasetC(Dataset):
    def __init__(self, csv_file, root_dir, main_dir, transform=None):
        self.root_dir = root_dir
        self.main_dir = main_dir
        self.csv_file = csv_file
        self.annotations = pd.read_csv(os.path.join(self.main_dir, self.csv_file))
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations['Image'][index])
        im = Image.open(img_path)
        image = T.functional.to_tensor(im)
        if(self.transform):
            image = self.transform(image)
        label = self.annotations['Label'][index]
        return image, label