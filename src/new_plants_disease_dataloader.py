from datasets.class_dataset import PlantImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import yaml

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
with open("/Users/aravdhoot/Plant_Disease_Classification/config.yaml", 'r') as yaml_file:
    parse_yaml = yaml.safe_load(yaml_file)

main_dir = parse_yaml['main_dir']['new_plants_disease']

train_dataset = PlantImageDataset(csv_file=parse_yaml['csv']['new_plants_disease']['train'], root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/New Plants Diseases Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train", main_dir=main_dir, transform=transforms)
valid_dataset = PlantImageDataset(csv_file=parse_yaml['csv']['new_plants_disease']['valid'], root_dir="/Users/aravdhoot/Plant_Disease_Classification/data/New Plants Diseases Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid", main_dir=main_dir, transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)