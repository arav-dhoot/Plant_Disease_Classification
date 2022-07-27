from datasets.class_dataset import PlantImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import yaml

with open("/Users/aravdhoot/Plant_Disease_Classification/config.yaml", 'r') as yaml_file:
    parse_yaml = yaml.safe_load(yaml_file)

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
main_dir = parse_yaml['main_dir']['plant_disease_recognition']

train_dataset = PlantImageDataset(csv_file=parse_yaml['csv']['plant_disease_recognition']['train'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['train'], main_dir=main_dir, transform=transforms)
valid_dataset = PlantImageDataset(csv_file=parse_yaml['csv']['plant_disease_recognition']['valid'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['valid'], main_dir=main_dir, transform=transforms)
test_dataset = PlantImageDataset(csv_file=parse_yaml['csv']['plant_disease_recognition']['test'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['test'], main_dir=main_dir, transform=transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)