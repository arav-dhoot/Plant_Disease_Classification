import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class PlantImageDataset(Dataset):
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
        if(self.transform):
            im = self.transform(im)
        label = self.annotations.iloc[index, 1]
        return im, label