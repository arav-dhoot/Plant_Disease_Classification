import argparse
from datasets import class_dataset_A, class_dataset_B, class_dataset_C
from torch.utils.data import DataLoader
import torchvision.transforms as T
import yaml
import torch
import pytorch_lightning as pl
from model.lightning_module import LitModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help='Either new_plants, plant_disease, or plant_pathology', choices=['new_plants', 'plant_disease', 'plant_pathology'], default='new_plants')
parser.add_argument("--epochs", help='Number of epochs', default=1, type=int)
parser.add_argument("--model", help="Either ResNet or ViT", choices=['ResNet', 'ViT'], default='ViT')
args = parser.parse_args()

BATCH_SIZE = 64
EPOCHS = args.epochs
NUM_WORKERS = 10
MODEL = args.model

transforms = T.Compose([T.Resize(size=(224, 224))])
with open('../config/config.yaml', 'r') as yaml_file:
    parse_yaml = yaml.safe_load(yaml_file)

DATASET = args.dataset
NUM_CLASSES = parse_yaml['num_classes'][DATASET]

if DATASET == "new_plants":

    main_dir = parse_yaml['main_dir']['new_plants_disease']

    train_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv']['new_plants_disease']['train'], root_dir=parse_yaml['root_dir']['new_plants_disease']['train'], main_dir=main_dir, transform=transforms)
    valid_dataset = class_dataset_A.PlantImageDatasetA(csv_file=parse_yaml['csv']['new_plants_disease']['valid'], root_dir=parse_yaml['root_dir']['new_plants_disease']['valid'], main_dir=main_dir, transform=transforms)

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [round(0.8 * len(train_dataset)), round(0.2 * len(train_dataset))])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
elif DATASET == 'plant_disease':

    main_dir = parse_yaml['main_dir']['plant_disease_recognition']

    train_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv']['plant_disease_recognition']['train'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['train'], main_dir=main_dir, transform=transforms)
    valid_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv']['plant_disease_recognition']['valid'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['valid'], main_dir=main_dir, transform=transforms)
    test_dataset = class_dataset_B.PlantImageDatasetB(csv_file=parse_yaml['csv']['plant_disease_recognition']['test'], root_dir=parse_yaml['root_dir']['plant_disease_recognition']['test'], main_dir=main_dir, transform=transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

elif DATASET == 'plant_pathology':

    main_dir = parse_yaml['main_dir']['plant_pathology']

    dataset = class_dataset_C.PlantImageDatasetC(csv_file=parse_yaml['csv']['plant_pathology'], root_dir=parse_yaml['root_dir']['plant_pathology'], main_dir=main_dir, transform=transforms)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [round(0.7 * len(dataset)), round(0.2 * len(dataset)), round(0.1 * len(dataset))])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) 

model = LitModel(NUM_CLASSES, MODEL)
trainer = pl.Trainer(max_epochs=EPOCHS)
trainer.fit(model, train_loader, valid_loader)
test_results = trainer.test(dataloaders=test_loader)
print(test_results)