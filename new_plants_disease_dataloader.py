from class_dataset import PlantImageDataset
from torch.utils.data import DataLoader

batch_size = 64
main_dir = "/Users/aravdhoot/CS_Research_Project/data/New Plants Diseases Dataset"

train_dataset = PlantImageDataset(csv_file= "new_plants_disease_train.csv", root_dir="/Users/aravdhoot/CS_Research_Project/data/New Plants Diseases Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train", main_dir=main_dir)
valid_dataset = PlantImageDataset(csv_file= "new_plants_disease_valid.csv", root_dir="/Users/aravdhoot/CS_Research_Project/data/New Plants Diseases Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid", main_dir=main_dir)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
