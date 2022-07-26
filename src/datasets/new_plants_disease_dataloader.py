from class_dataset import PlantImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
main_dir = "/Users/aravdhoot/Plant_Disease_Classification/data/New Plants Diseases Dataset"

train_dataset = PlantImageDataset(csv_file= "new_plants_disease_train.csv", root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/New Plants Diseases Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train", main_dir=main_dir, transform=transforms)
valid_dataset = PlantImageDataset(csv_file= "new_plants_disease_valid.csv", root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/New Plants Diseases Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid", main_dir=main_dir, transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)