import argparse
from datasets import class_dataset_A, class_dataset_B, class_dataset_C
from torch.utils.data import DataLoader
import torchvision.transforms as T
import yaml
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help='Either new_plants, plant_disease, or plant_pathology', choices=['new_plants', 'plant_disease', 'plant_pathology'], default='new_plants')
args = parser.parse_args()

batch_size = 64
transforms = T.Compose([T.Resize(size=(224, 224))])
with open('../config/config.yaml', 'r') as yaml_file:
    parse_yaml = yaml.safe_load(yaml_file)



if args.dataset == "new_plants":

    main_dir = parse_yaml['main_dir']['new_plants_disease']

    train_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv']['new_plants_disease']['train'], root_dir=parse_yaml['root_dir']['new_plants_disease']['train'], main_dir=main_dir, transform=transforms)
    valid_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv']['new_plants_disease']['valid'], root_dir=parse_yaml['root_dir']['new_plants_disease']['valid'], main_dir=main_dir, transform=transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

elif args.dataset == 'plant_disease':

    main_dir = parse_yaml['main_dir']['plant_disease_recognition']

    train_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv']['plant_disease_recognition']['train'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['train'], main_dir=main_dir, transform=transforms)
    valid_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv']['plant_disease_recognition']['valid'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['valid'], main_dir=main_dir, transform=transforms)
    test_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv']['plant_disease_recognition']['test'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['test'], main_dir=main_dir, transform=transforms)


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

elif args.dataset == 'plant_pathology':

    main_dir = parse_yaml['main_dir']['plant_pathology']

    dataset = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv']['plant_pathology'], root_dir=parse_yaml['root_dir']['plant_pathology'], main_dir=main_dir, transform=transforms)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images, labels)