import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np

class PlantImageDatasetB(Dataset):
    def __init__(self, csv_file, root_dir, main_dir, transform=None):
        self.root_dir = root_dir
        self.main_dir = main_dir
        self.csv_file = csv_file
        self.annotations = pd.read_csv(os.path.join(self.main_dir, self.csv_file))
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, f"{self.annotations.iloc[index, 1]}/{self.annotations.iloc[index, 0]}")
        im = read_image(img_path)
        image = T.functional.to_tensor(im.iloc[index, 1:-1].values.astype(np.uint8).reshape((1, 16, 16)))
        if(self.transform):
            image = self.transform(image)
        label = self.annotations.iloc[index, 1]
        return image, label