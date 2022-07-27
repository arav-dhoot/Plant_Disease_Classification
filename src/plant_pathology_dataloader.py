from datasets.plant_pathology_class_dataset import PlantImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch 
import yaml

with open("/Users/aravdhoot/Plant_Disease_Classification/config.yaml", 'r') as yaml_file:
    parse_yaml = yaml.safe_load(yaml_file)

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
main_dir = parse_yaml['main_dir']['plant_pathology']

dataset = PlantImageDataset(csv_file=parse_yaml['csv']['plant_pathology'], root_dir=parse_yaml['root_dir']['plant_pathology'], main_dir=main_dir, transform=transforms)

train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)