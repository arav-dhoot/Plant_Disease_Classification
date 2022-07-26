from plant_pathology_class_dataset import PlantImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch 

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
main_dir = "/Users/aravdhoot/Plant_Disease_Classification/data/plant-pathology-2020-fgvc7"

dataset = PlantImageDataset(csv_file="plant_pathology_train.csv", root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/plant-pathology-2020-fgvc7/Train", main_dir=main_dir, transform=transforms)

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)